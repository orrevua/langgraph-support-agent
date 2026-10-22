import pytest
from fastapi.testclient import TestClient

from supportagent.api import create_app

ANA = {"X-Customer-Id": "C001"}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, setup_logging=False)) as c:
        yield c


def start_refund(client, thread="t-refund"):
    r = client.post("/chat", json={"message": "Please refund TX-1003, it failed", "thread_id": thread}, headers=ANA)
    assert r.status_code == 200, r.text
    return r.json()


def test_chat_completes(client):
    body = client.post("/chat", json={"message": "What's my balance?"}, headers=ANA).json()
    assert body["status"] == "completed"
    assert "R$ 2.350,75" in body["reply"]


def test_refund_pauses_then_approve(client):
    body = start_refund(client)
    assert body["status"] == "pending_approval"
    action = body["pending"][0]
    assert action["tool"] == "issue_refund"
    assert action["args"]["amount_brl"] == 1200.0
    assert action["context"]["eligibility"]["refundable"] is True

    assert client.get("/threads/t-refund/pending").json()["pending"][0]["action_id"] == action["action_id"]
    # A new chat message is refused while the decision is pending
    assert client.post("/chat", json={"message": "hello?", "thread_id": "t-refund"}, headers=ANA).status_code == 409

    done = client.post("/threads/t-refund/approve", json={"reviewer": "maria.ops"}).json()
    assert done["status"] == "completed"
    assert "RF-" in done["reply"] and "R$ 1.200,00" in done["reply"]
    audit = client.get("/threads/t-refund/audit").json()
    assert audit[0]["decision"] == "approved" and audit[0]["reviewer"] == "maria.ops"
    assert client.get("/threads/t-refund/pending").json()["pending"] == []


def test_approve_with_edited_amount(client):
    start_refund(client, "t-edit")
    done = client.post(
        "/threads/t-edit/approve", json={"reviewer": "maria.ops", "edited_args": {"amount_brl": 500}}
    ).json()
    assert "R$ 500,00" in done["reply"]
    audit = client.get("/threads/t-edit/audit").json()[0]
    assert audit["proposed_args"]["amount_brl"] == 1200.0 and audit["final_args"]["amount_brl"] == 500


def test_edit_of_non_editable_arg_is_422(client):
    start_refund(client, "t-bad-edit")
    r = client.post("/threads/t-bad-edit/approve", json={"reviewer": "x", "edited_args": {"transaction_id": "TX-1001"}})
    assert r.status_code == 422


def test_reject(client):
    start_refund(client, "t-reject")
    done = client.post("/threads/t-reject/reject", json={"reviewer": "maria.ops", "reason": "suspicious"}).json()
    assert done["status"] == "completed"
    assert "not approved" in done["reply"]
    assert client.get("/threads/t-reject/audit").json()[0]["decision"] == "rejected"


def test_approve_without_pending_is_409(client):
    assert client.post("/threads/nope/approve", json={"reviewer": "x"}).status_code == 409


def test_other_customer_cannot_use_thread(client):
    start_refund(client, "t-owner")
    client.post("/threads/t-owner/reject", json={"reviewer": "x", "reason": "r"})
    r = client.post("/chat", json={"message": "hi", "thread_id": "t-owner"}, headers={"X-Customer-Id": "C002"})
    assert r.status_code == 404


def test_input_validation(client):
    assert client.post("/chat", json={"message": ""}, headers=ANA).status_code == 422
    assert client.post("/chat", json={"message": "x" * 2001}, headers=ANA).status_code == 422
    assert client.post("/chat", json={"message": "hi"}, headers={"X-Customer-Id": "1; drop"}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code == 422


def test_mock_payments_router(client):
    assert client.get("/mock-payments/v1/transactions/TX-1005/status").json()["network_status"] == "IN_PROCESS"
    assert client.get("/mock-payments/v1/transactions/TX-9999/status").status_code == 404
