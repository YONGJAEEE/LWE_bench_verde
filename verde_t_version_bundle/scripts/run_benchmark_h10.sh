#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE="${BUNDLE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
PROCESSED_DUMP_PATH="${PROCESSED_DUMP_PATH:-}"
DUMP_PATH="${DUMP_PATH:-$BUNDLE/generated/benchmark_h10}"
VERDE_OUTPUT_ROOT="${VERDE_OUTPUT_ROOT:-$BUNDLE/generated/verde_ready_h10}"
LOG_DIR="${LOG_DIR:-$BUNDLE/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/benchmark_h10_t6.log}"

if [[ -z "$PYTHON" ]]; then
  echo "No active Python found; set PYTHON or activate a Python environment." >&2
  exit 127
fi
if [[ -z "$PROCESSED_DUMP_PATH" ]]; then
  echo "Set PROCESSED_DUMP_PATH to the absolute processed benchmark directory." >&2
  exit 2
fi

export BUNDLE PYTHON
mkdir -p "$DUMP_PATH" "$VERDE_OUTPUT_ROOT" "$LOG_DIR"

COMMAND=(
  "$PYTHON" "$BUNDLE/scripts/generate_and_convert.py"
  --dump_path "$DUMP_PATH"
  --verde_output_root "$VERDE_OUTPUT_ROOT"
  --processed_dump_path "$PROCESSED_DUMP_PATH"
  --hamming 10
  --t_version 6
  --num_secret_seeds 1
  --max_samples "${MAX_SAMPLES:-4010000}"
)

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  "${COMMAND[@]}" >"$LOG_FILE" 2>&1
  exit $?
fi

nohup "${COMMAND[@]}" >"$LOG_FILE" 2>&1 &
echo "Started Hamming-10 t_6 generation (pid=$!, log=$LOG_FILE)"
