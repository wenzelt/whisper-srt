.PHONY: install test test-all lint format fix clean

install:
	uv sync

test:
	uv run pytest -m "not integration"

test-all:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix . && uv run ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f .coverage
	rm -rf htmlcov
