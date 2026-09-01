#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ASSIGNMENT_DIR/.venv-mdpdf"
MD2PDF="$VENV_DIR/bin/md2pdf"

INPUT="${1:-"$ASSIGNMENT_DIR/Ans.md"}"
OUTPUT="${2:-"$ASSIGNMENT_DIR/Ans.pdf"}"

if [[ ! -x "$MD2PDF" ]]; then
  echo "md2pdf is not installed at $MD2PDF" >&2
  echo "Install it with:" >&2
  echo "  /opt/anaconda3/bin/python3 -m venv .venv-mdpdf" >&2
  echo "  .venv-mdpdf/bin/python -m pip install md-to-pdf-cli pymupdf" >&2
  echo "  PLAYWRIGHT_BROWSERS_PATH=.playwright-browsers .venv-mdpdf/bin/python -m playwright install chromium" >&2
  exit 1
fi

export PLAYWRIGHT_BROWSERS_PATH="$ASSIGNMENT_DIR/.playwright-browsers"

"$MD2PDF" convert "$INPUT" \
  -o "$OUTPUT" \
  --math \
  --embed-images \
  --page-size Letter \
  --margin 0.7in

echo "PDF written to: $OUTPUT"
