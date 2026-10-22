# Draft: "Let the agent investigate, make a human press the button"

*Draft post for orrevua.dev. Placeholders in [BRACKETS] must be filled from real-model runs.*

## The hook

A support agent for a bank has to do two very different jobs. It answers questions like "why didn't
my Pix arrive?", and it takes actions like refunding R$ 1.200,00. A wrong answer to the first is
embarrassing. A wrong action is a financial incident. I wanted to learn how teams build the second
kind safely, so I built a small LangGraph agent for a made-up Brazilian bank and made a human
approval step central to the design.

## What I built

- An explicit LangGraph `StateGraph` (guard → agent ⇄ tools, plus a `human_review` node). I didn't
  use `create_react_agent`, because I wanted to see every edge.
- Seven tools over a Postgres domain model: customers, accounts, cards, Pix transactions and refunds.
  There is also a mock payments API that can be configured to return 503s.
- Sensitive tools (`issue_refund`, `block_card`) pause the graph with `interrupt()`. A reviewer
  approves, edits or rejects through FastAPI, and every decision lands in an audit table.
- Guardrails: an injection heuristic guard, PII masking in logs, refund caps, per-turn step limits,
  retries with backoff.
- Ten scripted eval scenarios scored on trajectory, answer and database side effects.

[GIF: docs/demo.gif]

## Three things that surprised me

1. **`interrupt()` re-executes the node.** ... (write-up: audit-after-interrupt, idempotent insert)
2. **A reviewer edit is a message replacement.** ... (`add_messages` and message ids)
3. **The fake model was the most useful test tool.** ... (policy vs replay; offline CI)

## Results

| | offline scripted model | Claude ([LLM_MODEL]) |
|---|---|---|
| scenarios passed | 10/10 (pipeline check) | [X/10] |
| trajectory passes | 10/10 | [X/10] |
| answer passes | 10/10 | [X/10] |
| avg model calls per scenario | 2.8 | [N] |
| avg latency per scenario | <0.1 s | [N s] |
| injection scenario | blocked by guard (no model call) | [blocked / reached model?] |

[Discussion of real-model failures: which scenarios, why, what I changed in the prompt or tools.]

## What I'd do next

- An LLM-as-judge for answers, alongside the keyword checks
- A small reviewer UI (Vite + React)
- Tracing, so I can see the approve/reject ratio and where time goes

Code: github.com/orrevua/langgraph-support-agent
