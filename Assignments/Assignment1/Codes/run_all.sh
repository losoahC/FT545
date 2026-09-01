#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${1:-"$SCRIPT_DIR/results.txt"}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /opt/anaconda3/bin/python3 ]]; then
    PYTHON_BIN=/opt/anaconda3/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ft545_mpl}"
mkdir -p "$MPLCONFIGDIR"

: > "$OUTPUT_FILE"

echo "Using Python: $PYTHON_BIN" | tee -a "$OUTPUT_FILE"
echo "Writing combined output to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"

for script in ass1.py ass2.py ass3.py ass4.py ass5.py; do
  {
    echo
    echo "===== $script ====="
  } | tee -a "$OUTPUT_FILE"

  "$PYTHON_BIN" "$SCRIPT_DIR/$script" 2>&1 | tee -a "$OUTPUT_FILE"
done

echo | tee -a "$OUTPUT_FILE"
echo "All scripts completed." | tee -a "$OUTPUT_FILE"
