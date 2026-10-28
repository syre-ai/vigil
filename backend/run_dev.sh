#!/usr/bin/env bash
# Runs the Vigil FastAPI backend for local development:
# uvicorn on :8000 with autoreload. CORS allows http://localhost:5173
# (the Vite dev server) by default -- see backend/app/config.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "error: $PYTHON_BIN not found -- expected the project venv at <repo>/.venv" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
