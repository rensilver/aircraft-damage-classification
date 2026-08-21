#!/usr/bin/env bash
set -euo pipefail

echo "==> ruff check"
uv run ruff check src tests scripts

echo "==> ruff format --check"
uv run ruff format --check src tests scripts

echo "==> mypy"
uv run mypy

echo "==> pytest"
uv run pytest -m "not slow"

echo "All checks passed."
