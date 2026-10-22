from langgraph.checkpoint.memory import MemorySaver

from supportagent.agent.state import user_turn
from supportagent.llm import build_fake_model
from supportagent.runtime import build_agent


def tool_names(messages):
    return [c["name"] for m in messages if m.type == "ai" for c in m.tool_calls]


def make(settings, repo, payments, **kw):
    graph, _ = build_agent(settings, repo=repo, payments=payments, model=build_fake_model(), **kw)
    return graph


def test_balance_loop(settings, repo, payments):
    out = make(settings, repo, payments).invoke(user_turn("What's my balance?", "C001"))
    assert tool_names(out["messages"]) == ["get_account_summary", "draft_reply"]
    assert "R$ 2.350,75" in out["messages"][-1].content
    assert out["agent_steps"] == 3


def test_failed_pix_investigation(settings, repo, payments):
    out = make(settings, repo, payments).invoke(user_turn("My rent Pix never arrived", "C001"))
    assert tool_names(out["messages"]) == [
        "list_recent_transactions",
        "check_transaction_status",
        "draft_reply",
    ]
    assert "AC03" in out["messages"][-1].content


def test_out_of_scope_uses_no_tools(settings, repo, payments):
    out = make(settings, repo, payments).invoke(user_turn("Write me a poem about the sea", "C001"))
    assert tool_names(out["messages"]) == []


def test_step_counter_resets_each_turn(settings, repo, payments):
    graph = make(settings, repo, payments, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(user_turn("What's my balance?", "C001"), config)
    out = graph.invoke(user_turn("And my recent transactions?", "C001"), config)
    assert out["agent_steps"] == 3  # reset by RESET_STEPS, not 6


def test_step_limit_escalates(settings, repo, payments):
    settings.max_agent_steps = 1
    out = make(settings, repo, payments).invoke(user_turn("What's my balance?", "C001"))
    assert "escalated" in out["messages"][-1].content
