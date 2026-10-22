"""Unit tests for tool business logic (no graph, no model)."""

import json
import logging

import pytest

from supportagent.domain.seed import TRANSACTIONS
from supportagent.payments import MockPaymentsClient
from supportagent.tools import ToolContext, ToolError, build_tools, refund_eligibility, to_cents


@pytest.fixture
def tools(ctx):
    return {t.name: t for t in build_tools(ctx)}


def run(tools, name, **args):
    return json.loads(tools[name].invoke({"customer_id": "C001", **args}))


def txn(tid):
    return next(t for t in TRANSACTIONS if t.id == tid)


def test_injected_args_are_hidden_from_the_model(tools):
    for tool in tools.values():
        schema = tool.tool_call_schema.model_json_schema()
        assert "customer_id" not in schema.get("properties", {})
        assert "approved_by" not in schema.get("properties", {})


def test_account_summary(tools):
    data = run(tools, "get_account_summary")
    assert data["total_balance"] == "R$ 12.350,75"
    assert {c["last4"] for c in data["cards"]} == {"4242", "5555"}


def test_lookup_customer_masks_pii_and_checks_identifier(tools):
    data = run(tools, "lookup_customer", identifier="987.654.321-00")
    assert data["identifier_belongs_to_customer"] is False
    assert "123.456.789-09" not in json.dumps(data)
    assert run(tools, "lookup_customer", identifier="12345678909")["identifier_belongs_to_customer"] is True


def test_list_transactions_filters_and_validates(tools):
    failed = run(tools, "list_recent_transactions", status="failed")
    assert [t["transaction_id"] for t in failed] == ["TX-1006", "TX-1003"]
    with pytest.raises(ToolError):
        run(tools, "list_recent_transactions", limit=500)


def test_other_customers_transactions_are_invisible(tools):
    with pytest.raises(ToolError, match="not found on this customer's accounts"):
        run(tools, "check_transaction_status", transaction_id="TX-2001")


def test_check_status_reports_refund_eligibility(tools):
    data = run(tools, "check_transaction_status", transaction_id="tx-1003")
    assert data["network"]["network_status"] == "REJECTED"
    assert data["refund"]["refundable"] is True and data["refund"]["max_refund"] == "R$ 1.200,00"


def test_check_status_retries_transient_failures(repo, settings):
    payments = MockPaymentsClient(repo, fail_first=2)
    tools = {t.name: t for t in build_tools(ToolContext(repo, payments, settings))}
    assert run(tools, "check_transaction_status", transaction_id="TX-1005")["network"]["network_status"] == "IN_PROCESS"
    assert payments.calls == 3


def test_check_status_gives_up_gracefully(repo, settings):
    payments = MockPaymentsClient(repo, fail_first=10)
    tools = {t.name: t for t in build_tools(ToolContext(repo, payments, settings))}
    with pytest.raises(ToolError, match="temporarily unavailable"):
        run(tools, "check_transaction_status", transaction_id="TX-1005")
    assert payments.calls == settings.payments_retry_attempts


@pytest.mark.parametrize(
    ("tid", "refundable"),
    [("TX-1003", True), ("TX-1008", True), ("TX-1006", False), ("TX-1001", False), ("TX-1002", False)],
)
def test_refund_eligibility_rules(tid, refundable):
    assert refund_eligibility(txn(tid), [], 200_000)["refundable"] is refundable


def test_refund_eligibility_caps_at_hard_limit():
    result = refund_eligibility(txn("TX-1003"), [], 50_000)
    assert result["max_refund_cents"] == 50_000 and result["needs_escalation"] is True


def test_issue_refund_happy_path_credits_balance(tools, repo):
    data = run(tools, "issue_refund", approved_by="r", transaction_id="TX-1003", amount_brl=1200, reason="x")
    assert data["status"] == "refunded"
    assert repo.accounts["ACC-001"].balance_cents == 235_075 + 120_000


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"transaction_id": "TX-1003", "amount_brl": 1200.01}, "exceeds the refundable maximum"),
        ({"transaction_id": "TX-1003", "amount_brl": -5}, "must be positive"),
        ({"transaction_id": "TX-1003", "amount_brl": 5000}, "must be escalated"),
        ({"transaction_id": "TX-1001", "amount_brl": 10}, "not refundable"),
        ({"transaction_id": "TX-2002", "amount_brl": 10}, "not found"),
    ],
)
def test_issue_refund_limits(tools, args, error):
    with pytest.raises(ToolError, match=error):
        run(tools, "issue_refund", approved_by="r", reason="x", **args)


def test_refund_cannot_be_repeated(tools):
    run(tools, "issue_refund", approved_by="r", transaction_id="TX-1008", amount_brl=39.90, reason="dup")
    with pytest.raises(ToolError, match="already fully refunded"):
        run(tools, "issue_refund", approved_by="r", transaction_id="TX-1008", amount_brl=1, reason="dup")


def test_to_cents_rounding():
    assert to_cents(39.9) == 3990
    assert to_cents("0.005") == 1
    with pytest.raises(ToolError):
        to_cents("abc")


def test_block_card(tools, repo):
    assert run(tools, "block_card", approved_by="r", card_id="CARD-001", reason="lost")["status"] == "blocked"
    assert run(tools, "block_card", approved_by="r", card_id="CARD-001", reason="lost")["status"] == "already_blocked"
    with pytest.raises(ToolError):
        run(tools, "block_card", approved_by="r", card_id="CARD-003", reason="lost")  # Bruno's card


def test_draft_reply_uses_first_name(tools):
    assert run(tools, "draft_reply", message="Done.")["reply"].startswith("Hi Ana,")


def test_tools_node_turns_exceptions_into_error_messages(ctx, caplog):
    from langchain_core.messages import AIMessage

    from supportagent.agent.graph import make_tools_node

    node = make_tools_node(build_tools(ctx))
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "check_transaction_status", "args": {"transaction_id": "TX-9999"}, "id": "a"},
            {"name": "list_recent_transactions", "args": {"limit": "lots"}, "id": "b"},
            {"name": "no_such_tool", "args": {}, "id": "c"},
        ],
    )
    with caplog.at_level(logging.ERROR):
        out = node({"messages": [ai], "customer_id": "C001"})["messages"]
    assert [m.status for m in out] == ["error", "error", "error"]
    assert "not found" in out[0].content and "Internal error" in out[1].content and "Unknown tool" in out[2].content


def test_indirect_injection_in_tool_output_is_flagged(repo, payments, settings):
    tools = {t.name: t for t in build_tools(ToolContext(repo, payments, settings))}
    from langchain_core.messages import AIMessage

    from supportagent.agent.graph import make_tools_node

    node = make_tools_node(list(tools.values()))
    ai = AIMessage("", tool_calls=[{"name": "list_recent_transactions", "args": {}, "id": "a"}])
    out = node({"messages": [ai], "customer_id": "C003"})["messages"][0]
    assert out.content.startswith("[SECURITY NOTE")
