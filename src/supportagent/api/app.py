"""FastAPI app: chat endpoint + reviewer endpoints for human-in-the-loop approvals.

    uv run uvicorn supportagent.api:create_app --factory --reload

Authentication is mocked: the customer is identified by the `X-Customer-Id` header (in a real
system this comes from a verified session/JWT, never from the request body). Reviewer
endpoints would sit behind back-office auth.
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack, asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from supportagent.agent.review import PendingAction, ReviewDecision
from supportagent.checkpointing import open_checkpointer
from supportagent.config import Settings, get_settings
from supportagent.domain.models import ApprovalRecord
from supportagent.payments import MockPaymentsClient, build_payments_router
from supportagent.repository import build_repository
from supportagent.runtime import build_agent
from supportagent.service import AgentService, InvalidEdit, NothingPending, ThreadBusy, ThreadNotFound, TurnResult

THREAD_ID = r"^[A-Za-z0-9_-]{1,64}$"
CustomerId = Annotated[str, Header(alias="X-Customer-Id", pattern=r"^C\d{3,10}$")]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID)


class ApproveRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    edited_args: dict[str, Any] | None = None
    note: str | None = Field(default=None, max_length=500)


class RejectRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)


class PendingResponse(BaseModel):
    thread_id: str
    pending: list[PendingAction]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with ExitStack() as stack:
            repo = build_repository(settings)
            if hasattr(repo, "close"):
                stack.callback(repo.close)
            checkpointer = stack.enter_context(open_checkpointer(settings.storage, settings.database_url))
            graph, ctx = build_agent(settings, repo=repo, checkpointer=checkpointer)
            app.state.service = AgentService(graph, settings)
            app.state.repo = repo
            # The mock payments API is served by the same app (for HttpPaymentsClient demos).
            app.include_router(build_payments_router(MockPaymentsClient(repo)), prefix="/mock-payments")
            yield

    app = FastAPI(title="Aurora Bank support agent", version="0.1.0", lifespan=lifespan)

    def service() -> AgentService:
        return app.state.service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "storage": settings.storage, "provider": settings.resolved_provider}

    @app.post("/chat", response_model=TurnResult)
    def chat(body: ChatRequest, customer_id: CustomerId) -> TurnResult:
        thread_id = body.thread_id or uuid.uuid4().hex
        try:
            return service().chat(thread_id, customer_id, body.message)
        except ThreadNotFound as exc:
            raise HTTPException(404, "thread not found") from exc
        except ThreadBusy as exc:
            raise HTTPException(409, "thread is waiting for a reviewer decision") from exc

    @app.get("/threads/{thread_id}/pending", response_model=PendingResponse)
    def pending(thread_id: str) -> PendingResponse:
        return PendingResponse(thread_id=thread_id, pending=service().pending(thread_id))

    def decide(thread_id: str, decision: ReviewDecision) -> TurnResult:
        try:
            return service().decide(thread_id, decision)
        except NothingPending as exc:
            raise HTTPException(409, "no pending action on this thread") from exc
        except InvalidEdit as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/threads/{thread_id}/approve", response_model=TurnResult)
    def approve(thread_id: str, body: ApproveRequest) -> TurnResult:
        return decide(
            thread_id,
            ReviewDecision(decision="approve", reviewer=body.reviewer, edited_args=body.edited_args, reason=body.note),
        )

    @app.post("/threads/{thread_id}/reject", response_model=TurnResult)
    def reject(thread_id: str, body: RejectRequest) -> TurnResult:
        return decide(thread_id, ReviewDecision(decision="reject", reviewer=body.reviewer, reason=body.reason))

    @app.get("/threads/{thread_id}/audit", response_model=list[ApprovalRecord])
    def audit(thread_id: str) -> list[ApprovalRecord]:
        return app.state.repo.list_approvals(thread_id)

    return app
