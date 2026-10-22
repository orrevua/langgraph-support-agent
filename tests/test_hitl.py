import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from supportagent.agent.state import user_turn
from supportagent.checkpointing import open_checkpointer
from supportagent.llm import build_fake_model
from supportagent.runtime import build_agent


def agent(settings, repo, payments, checkpointer):
    graph, _ = build_agent(settings, repo=repo, payments=payments, model=build_fake_model(), checkpointer=checkpointer)
    return graph


def test_graph_pauses_before_sensitive_tool(settings, repo, payments):
    graph = agent(settings, repo, payments, MemorySaver())
    config = {"configurable": {"thread_id": "h1"}}
    graph.invoke(user_turn("I lost my debit card, block it", "C001"), config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.interrupts[0].value["tool"] == "block_card"
    assert repo.get_card("CARD-001").status == "active"  # nothing happened yet

    graph.invoke(Command(resume={"decision": "approve", "reviewer": "r1"}), config)
    assert repo.get_card("CARD-001").status == "blocked"
    assert repo.list_approvals("h1")[0].decision == "approved"


def test_rejection_reaches_the_model(settings, repo, payments):
    graph = agent(settings, repo, payments, MemorySaver())
    config = {"configurable": {"thread_id": "h2"}}
    graph.invoke(user_turn("Refund TX-1008 please, I was charged twice", "C001"), config)
    out = graph.invoke(Command(resume={"decision": "reject", "reviewer": "r1", "reason": "valid charge"}), config)
    rejected = [m for m in out["messages"] if m.type == "tool" and "rejected by reviewer" in str(m.content)]
    assert rejected and "not approved" in out["messages"][-1].content
    assert repo.list_refunds("TX-1008") == []


def test_tools_node_refuses_unapproved_sensitive_call(ctx, settings):
    from langchain_core.messages import AIMessage

    from supportagent.agent.graph import make_tools_node
    from supportagent.tools import build_tools

    node = make_tools_node(build_tools(ctx))
    ai = AIMessage(
        "", tool_calls=[{"name": "block_card", "args": {"card_id": "CARD-001", "reason": "lost"}, "id": "x"}]
    )
    out = node({"messages": [ai], "customer_id": "C001"})
    assert "requires human approval" in out["messages"][0].content
    assert ctx.repo.get_card("CARD-001").status == "active"


@pytest.mark.postgres
def test_pending_approval_survives_restart(database_url, pg_repo, settings, payments):
    thread = {"configurable": {"thread_id": f"restart-{uuid.uuid4()}"}}
    with open_checkpointer("postgres", database_url) as saver:
        agent(settings, pg_repo, payments, saver).invoke(user_turn("Please refund TX-1003", "C001"), thread)
    with open_checkpointer("postgres", database_url) as saver:  # "new process"
        graph = agent(settings, pg_repo, payments, saver)
        assert graph.get_state(thread).interrupts
        graph.invoke(Command(resume={"decision": "approve", "reviewer": "ops"}), thread)
    assert [r.amount_cents for r in pg_repo.list_refunds("TX-1003")] == [120_000]
    audit = pg_repo.list_approvals(thread["configurable"]["thread_id"])
    assert audit[0].decision == "approved" and audit[0].reviewer == "ops"
