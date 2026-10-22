.PHONY: setup test lint fmt demo db-seed
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
