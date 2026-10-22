.PHONY: setup test lint fmt db-seed api demo demo-interactive demo-postgres eval eval-postgres graph gif
# Offline by default; export LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY to use Claude.
export LLM_PROVIDER ?= fake

setup:            ## install deps and seed the local Postgres (p2_agent)
	uv sync
	-uv run python -m supportagent.db seed

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

db-seed:
	uv run python -m supportagent.db seed

api:
	uv run uvicorn supportagent.api:create_app --factory --reload --port 8000

demo:             ## approval-flow walkthrough (offline, in-memory)
	uv run python -m supportagent.demo

demo-interactive: ## you are the reviewer
	uv run python -m supportagent.demo --interactive

demo-postgres: db-seed
	uv run python -m supportagent.demo --storage postgres

eval:
	uv run python -m supportagent.evals --report reports/eval-results.md

eval-postgres: db-seed
	uv run python -m supportagent.evals --storage postgres --report reports/eval-results-postgres.md

graph:            ## regenerate docs/graph.mmd from the compiled graph
	uv run python scripts/render_graph.py

gif:              ## needs vhs installed
	vhs docs/demo.tape
