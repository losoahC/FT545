#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$REPO_DIR/venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/venv/bin/python"
else
  PYTHON=python3
fi
"$PYTHON" "$SCRIPT_DIR/run_assignment.py"
"$PYTHON" "$SCRIPT_DIR/validate.py"
