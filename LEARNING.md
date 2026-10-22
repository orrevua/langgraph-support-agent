# Learning guide

A study path through this repo, in order. Each step has a file to read, a command to run, and the
concept it teaches. Plan on 4–6 hours for the full path.

## Study path

| # | Read | Run | Concept |
|---|------|-----|---------|
| 1 | `examples/01_fundamentals.py` | `uv run python examples/01_fundamentals.py` | Chat models, messages, `ChatPromptTemplate`, LCEL pipes, structured output with Pydantic, `@tool` and the tool-calling round trip |
| 2 | `src/supportagent/llm/__init__.py`, `llm/fake.py` | `uv run pytest tests/test_fake_model.py` | Programming against `BaseChatModel`; how `bind_tools` / `with_structured_output` work under the hood; building a deterministic fake |
| 3 | `src/supportagent/tools.py` | `uv run pytest tests/test_tools.py -v` | Tool design: docstrings as model-facing docs, `InjectedToolArg`, errors as values, business rules in code |
| 4 | `src/supportagent/agent/state.py` | | Typed state, **reducers** (`add_messages`, custom `count_steps`, `merge_dicts`) |
| 5 | `src/supportagent/agent/graph.py` | `uv run python examples/02_first_graph.py` | A hand-written ReAct loop: nodes, conditional edges, the tools node, step limits |
| 6 | `src/supportagent/llm/policy.py` | | How a "planner" decides the next action; compare it with the real system prompt in `agent/prompts.py` |
| 7 | `src/supportagent/checkpointing.py` | `uv run python examples/03_time_travel.py` | Checkpointers, `thread_id`, `get_state_history`, replay and fork |
| 8 | `src/supportagent/agent/review.py` | `make demo` then `make demo-interactive` | `interrupt()`, `Command(resume=...)`, editing a tool call, why side effects go after interrupts |
| 9 | `src/supportagent/service.py`, `api/app.py` | `make api` + the curl calls in the README | Exposing a paused graph over HTTP; ownership checks; 409 while a decision is pending |
| 10 | `src/supportagent/guardrails.py`, `resilience.py`, `payments.py` | `uv run pytest tests/test_guardrails.py tests/test_resilience.py` | Defense in depth: injection heuristics, PII masking in logs, retries with backoff, graceful tool errors |
| 11 | `src/supportagent/evals.py`, `evals/scenarios.yaml` | `make eval` | Agent evals: trajectory vs answer vs side-effect scoring; designing scenarios that also work for a real model |

## Official docs

- LangChain concepts (models, messages, tools, structured output): https://python.langchain.com/docs/concepts/
- LangGraph low-level concepts (state, reducers, nodes, edges): https://langchain-ai.github.io/langgraph/concepts/low_level/
- Persistence and checkpointers: https://langchain-ai.github.io/langgraph/concepts/persistence/
- Time travel: https://langchain-ai.github.io/langgraph/concepts/time-travel/
- Human-in-the-loop and `interrupt`: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- `langgraph-checkpoint-postgres`: https://pypi.org/project/langgraph-checkpoint-postgres/
- Anthropic tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Prompt injection background (OWASP LLM01): https://genai.owasp.org/llmrisk/llm01-prompt-injection/

## Improve it yourself

1. **[easy] Run the evals on Claude.** Set `ANTHROPIC_API_KEY` and run `make eval LLM_PROVIDER=anthropic`.
   Which scenarios fail? Is the cause the prompt, a tool docstring, or a scoring rule that is too
   strict? Commit the report next to the offline one.
2. **[easy] Add a scenario** for a refund above the hard limit (add a large failed Pix to
   `domain/seed.py`). Expected behavior: the agent explains it must be escalated, and there is no
   interrupt.
3. **[easy] Add a read-only tool** `get_pix_limits` (daytime/nighttime limits). Update the fake
   policy and add a scenario for it.
4. **[medium] Write an LLM-as-judge.** Add an optional `--judge` flag to `evals.py` that uses
   `LLM_FAST_MODEL` with `with_structured_output` to grade answers against a rubric (correct,
   polite, no hallucinated amounts). Compare it with the keyword checks.
5. **[medium] Stream tokens to the client.** Add `POST /chat/stream` (SSE) using
   `graph.stream(..., stream_mode=["messages", "updates"])`, and emit an event when the graph pauses.
6. **[medium] Use a classifier for injection.** Replace or combine the regex guard with a
   fast-model classifier, and measure false positives on 50 benign messages you write yourself.
7. **[medium] Approval policies.** Auto-approve refunds under R$ 50 for premium customers. Keep
   every decision in the audit table with `reviewer="policy:auto"`. Where does this logic belong:
   the graph, the service, or the tool?
8. **[medium] Use the async API.** Move to `AsyncPostgresSaver`, `ainvoke` and async FastAPI
   endpoints. Measure throughput with 50 concurrent chats against the mock payments API with
   `PAYMENTS_LATENCY_MS=300`.
9. **[hard] Add multiple pending actions to the API.** If the model proposes a refund *and* a card block in
   one message, the API currently returns them one at a time. Change it to accept a batch of
   decisions keyed by `action_id` (look up resuming several interrupts with a map of interrupt ids).
10. **[hard] Add a reviewer UI.** Build a small Vite + React + TS page that lists threads with pending
    actions, shows the `context` payload, and approves, edits or rejects them.
11. **[hard] Add observability.** Add OpenTelemetry or LangSmith tracing, then build a dashboard of tool
    error rates, interrupts per thread, approve/reject ratio, and p95 latency per node.
12. **[hard] Add long-term memory.** Use a LangGraph `Store` to remember customer preferences (language,
    preferred channel) across threads, with PII rules for what may be stored.
