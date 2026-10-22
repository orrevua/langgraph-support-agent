import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from supportagent.agent.state import user_turn
from supportagent.checkpointing import open_checkpointer
from supportagent.llm import build_fake_model
from supportagent.runtime import build_agent


def agent(settings, repo, payments, checkpointer):
    graph, _ = build_agent(settings, repo=repo, payments=payments, model=build_fake_model(), checkpointer=checkpointer)
    return graph


def test_threads_are_isolated(settings, repo, payments):
    graph = agent(settings, repo, payments, MemorySaver())
    a = {"configurable": {"thread_id": "a"}}
    b = {"configurable": {"thread_id": "b"}}
    graph.invoke(user_turn("What's my balance?", "C001"), a)
    graph.invoke(user_turn("What's my balance?", "C001"), a)
    graph.invoke(user_turn("Hi", "C001"), b)
    assert len(graph.get_state(a).values["messages"]) == 12
    assert len(graph.get_state(b).values["messages"]) == 2


def test_fork_from_earlier_checkpoint(settings, repo, payments):
    graph = agent(settings, repo, payments, MemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke(user_turn("What's my balance?", "C001"), config)
    graph.invoke(user_turn("List my recent transactions", "C001"), config)
    history = list(graph.get_state_history(config))
    end_of_turn_1 = next(s for s in reversed(history) if not s.next and s.values.get("messages"))
    fork = graph.invoke(user_turn("Hi", "C001"), end_of_turn_1.config)
    humans = [m.content for m in fork["messages"] if m.type == "human"]
    assert humans == ["What's my balance?", "Hi"]
    # the original branch is still reachable through history
    assert len(list(graph.get_state_history(config))) > len(history)


@pytest.mark.postgres
def test_postgres_checkpointer_roundtrip(database_url, settings, repo, payments):
    thread = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    with open_checkpointer("postgres", database_url) as saver:
        agent(settings, repo, payments, saver).invoke(user_turn("What's my balance?", "C001"), thread)
    # A brand-new saver (think: API restart) sees the same conversation.
    with open_checkpointer("postgres", database_url) as saver:
        state = agent(settings, repo, payments, saver).get_state(thread)
        assert "R$ 2.350,75" in state.values["messages"][-1].content


@pytest.mark.postgres
def test_postgres_repository_matches_memory(pg_repo, repo):
    assert pg_repo.get_customer("C001") == repo.get_customer("C001")
    assert [t.id for t in pg_repo.list_transactions("C001", limit=5)] == [
        t.id for t in repo.list_transactions("C001", limit=5)
    ]
    assert [t.id for t in pg_repo.list_transactions("C001", status="failed")] == ["TX-1006", "TX-1003"]
    refund = pg_repo.create_refund(
        transaction_id="TX-1003", customer_id="C001", amount_cents=1000, reason="t", approved_by="x"
    )
    assert refund.id.startswith("RF-")
    assert pg_repo.list_accounts("C001")[0].balance_cents == 235_075 + 1000
    assert pg_repo.set_card_status("CARD-001", "blocked").status == "blocked"
