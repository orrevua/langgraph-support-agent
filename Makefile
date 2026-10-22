.PHONY: setup test lint fmt demo db-seed api eval eval-postgres
export LLM_PROVIDER ?= fake

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	uv run python examples/01_fundamentals.py

db-seed:
	uv run python -m supportagent.db seed

api:
	uv run uvicorn supportagent.api:create_app --factory --reload --port 8000

eval:
	uv run python -m supportagent.evals --report reports/eval-results.md

eval-postgres: db-seed
	uv run python -m supportagent.evals --storage postgres --report reports/eval-results-postgres.md
