"""Scenario-based agent evaluation: `uv run python -m supportagent.evals`.

For each scenario in `evals/scenarios.yaml`:
  1. build a fresh agent (fresh seeded data, fresh checkpointer, configured payments chaos),
  2. send the user message, answering every approval interrupt with the scripted decision,
  3. score the trajectory (tool calls, interrupts, guardrail flags), the final answer and the
     side effects in the database.

Works with the offline fake (default) and with Claude (`LLM_PROVIDER=anthropic`). Results are
printed as a table and written to `reports/` with the provider name, so numbers are never
mixed up between models.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from supportagent.agent.review import ReviewDecision
from supportagent.checkpointing import open_checkpointer
from supportagent.config import Settings, get_settings
from supportagent.guardrails import configure_logging
from supportagent.llm import describe_provider, get_chat_model
from supportagent.payments import MockPaymentsClient
from supportagent.repository import BankRepository, InMemoryBankRepository
from supportagent.runtime import build_agent
from supportagent.service import AgentService

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScenarioResult:
    id: str
    category: str
    tools: list[str]
    interrupts: list[str]
    answer: str
    trajectory_ok: bool
    answer_ok: bool
    effects_ok: bool
    failures: list[str] = field(default_factory=list)
    model_calls: int = 0
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.trajectory_ok and self.answer_ok and self.effects_ok


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(any(name == a for a in it) for name in expected)


def score_trajectory(expect: dict, tools: list[str], interrupts: list[str], flags: list[str]) -> list[str]:
    failures = []
    wanted = expect.get("tools_in_order", [])
    if not is_subsequence(wanted, tools):
        failures.append(f"expected tools in order {wanted}, got {tools}")
    if bad := sorted(set(tools) & set(expect.get("forbidden_tools", []))):
        failures.append(f"forbidden tools called: {bad}")
    if interrupts != expect.get("interrupts", []):
        failures.append(f"expected interrupts {expect.get('interrupts', [])}, got {interrupts}")
    if missing := set(expect.get("flags", [])) - set(flags):
        failures.append(f"missing guardrail flags {sorted(missing)}")
    return failures


def score_answer(expect: dict, answer: str) -> list[str]:
    failures = []
    lower = answer.lower()
    for group in expect.get("answer_contains_any", []):
        if not any(option.lower() in lower for option in group):
            failures.append(f"answer lacks any of {group}")
    for banned in expect.get("answer_not_contains", []):
        if banned.lower() in lower:
            failures.append(f"answer contains {banned!r}")
    return failures


def score_effects(expect: dict, repo: BankRepository) -> list[str]:
    failures = []
    effects = expect.get("effects", {})
    if spec := effects.get("refund_created"):
        refunds = [r for r in repo.list_refunds(spec["transaction_id"]) if r.approved_by != "system"]
        if not any(r.amount_cents == spec["amount_cents"] for r in refunds):
            failures.append(f"no refund of {spec['amount_cents']} cents for {spec['transaction_id']}")
    spec = effects.get("no_refund")
    if spec and [r for r in repo.list_refunds(spec["transaction_id"]) if r.approved_by != "system"]:
        failures.append(f"unexpected refund for {spec['transaction_id']}")
    if spec := effects.get("card_status"):
        card = repo.get_card(spec["card_id"])
        if card is None or card.status != spec["status"]:
            failures.append(f"card {spec['card_id']} is {card.status if card else 'missing'}, want {spec['status']}")
    return failures


def run_scenario(scenario: dict, settings: Settings, storage: str) -> ScenarioResult:
    with ExitStack() as stack:
        if storage == "postgres":
            from supportagent.repository.postgres import PostgresBankRepository

            repo: BankRepository = PostgresBankRepository(settings.database_url)
            stack.callback(repo.close)
            repo.reset_and_seed()
        else:
            repo = InMemoryBankRepository()
        checkpointer = stack.enter_context(open_checkpointer(storage, settings.database_url))
        payments = MockPaymentsClient(repo, **scenario.get("payments", {}))
        graph, _ = build_agent(
            settings, repo=repo, payments=payments, model=get_chat_model(settings), checkpointer=checkpointer
        )
        service = AgentService(graph, settings)
        thread_id = f"eval-{scenario['id']}-{uuid.uuid4().hex[:6]}"

        started = time.perf_counter()
        approvals = list(scenario.get("approvals", []))
        interrupts: list[str] = []
        failures: list[str] = []
        result = service.chat(thread_id, scenario["customer_id"], scenario["message"])
        while result.status == "pending_approval":
            interrupts.append(result.pending[0].tool)
            if approvals:
                decision = ReviewDecision.model_validate(approvals.pop(0))
            else:
                failures.append(f"unexpected approval request for {result.pending[0].tool}")
                decision = ReviewDecision(decision="reject", reviewer="eval-runner", reason="unexpected action")
            result = service.decide(thread_id, decision)
        seconds = time.perf_counter() - started

        state = graph.get_state({"configurable": {"thread_id": thread_id}}).values
        tools = [c["name"] for m in state["messages"] if m.type == "ai" for c in m.tool_calls]
        model_calls = state.get("agent_steps", 0)  # LLM calls in this turn (guard replies excluded)
        answer = result.reply or ""
        expect = scenario["expect"]
        trajectory = score_trajectory(expect, tools, interrupts, state.get("flags", [])) + failures
        answer_failures = score_answer(expect, answer)
        effects = score_effects(expect, repo)
        return ScenarioResult(
            id=scenario["id"],
            category=scenario.get("category", ""),
            tools=tools,
            interrupts=interrupts,
            answer=answer,
            trajectory_ok=not trajectory,
            answer_ok=not answer_failures,
            effects_ok=not effects,
            failures=trajectory + answer_failures + effects,
            model_calls=model_calls,
            seconds=round(seconds, 2),
        )


def load_scenarios(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    defaults = data.get("defaults", {})
    return [{**defaults, **s} for s in data["scenarios"]]


def render(results: list[ScenarioResult], provider: str, storage: str, console: Console) -> None:
    table = Table(title=f"Agent evals - provider: {provider} - storage: {storage}")
    for col in ("scenario", "category", "trajectory", "answer", "effects", "tools called", "calls", "s"):
        table.add_column(col)
    mark = {True: "[green]PASS[/]", False: "[red]FAIL[/]"}
    for r in results:
        table.add_row(
            r.id,
            r.category,
            mark[r.trajectory_ok],
            mark[r.answer_ok],
            mark[r.effects_ok],
            " > ".join(r.tools) or "-",
            str(r.model_calls),
            f"{r.seconds:.2f}",
        )
    console.print(table)
    passed = sum(r.passed for r in results)
    console.print(f"[bold]{passed}/{len(results)} scenarios passed[/]")
    for r in results:
        for failure in r.failures:
            console.print(f"  [red]{r.id}[/]: {failure}")


def to_markdown(results: list[ScenarioResult], provider: str, storage: str, command: str = "make eval") -> str:
    ok = {True: "PASS", False: "**FAIL**"}
    lines = [
        "# Agent eval results",
        "",
        f"- Provider: **{provider}**",
        f"- Storage: {storage} (repository + checkpointer)",
        f"- Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `{command}`",
        f"- Result: **{sum(r.passed for r in results)}/{len(results)} scenarios passed**",
        "",
        "> With the offline scripted model this is a regression test of the whole pipeline (graph,",
        "> tools, approvals, guardrails, scoring), not a measure of LLM quality: the fake policy was",
        "> written for these scenarios. Re-run with `LLM_PROVIDER=anthropic` for real-model numbers.",
        "",
        "| scenario | category | trajectory | answer | effects | tools called | model calls | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.category} | {ok[r.trajectory_ok]} | {ok[r.answer_ok]} | {ok[r.effects_ok]} | "
            f"{' > '.join(r.tools) or '-'} | {r.model_calls} | {r.seconds:.2f} |"
        )
    failures = [f"- `{r.id}`: {f}" for r in results for f in r.failures]
    if failures:
        lines += ["", "## Failures", "", *failures]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent eval scenarios")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "evals" / "scenarios.yaml")
    parser.add_argument("--storage", choices=["memory", "postgres"], default="memory")
    parser.add_argument("--only", help="run a single scenario id")
    parser.add_argument("--report", type=Path, help="write markdown + json report to this path (.md)")
    args = parser.parse_args()

    configure_logging(logging.WARNING)  # PII-masked logs; shows retries and blocked injections
    base = get_settings()
    settings = Settings(**{**base.model_dump(), "storage": args.storage, "payments_retry_base_delay_s": 0.01})
    provider = describe_provider(settings)
    scenarios = [s for s in load_scenarios(args.scenarios) if not args.only or s["id"] == args.only]
    results = [run_scenario(s, settings, args.storage) for s in scenarios]

    render(results, provider, args.storage, Console(width=170))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        command = "make eval" if args.storage == "memory" else "make eval-postgres"
        args.report.write_text(to_markdown(results, provider, args.storage, command))
        payload = {"provider": provider, "storage": args.storage, "results": [asdict(r) for r in results]}
        args.report.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
