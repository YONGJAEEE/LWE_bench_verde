#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <verde-dataset-dir> <t-version> [additional train.py args ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE="${BUNDLE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VERDE_ROOT="${BUNDLE}/verde"
DATASET_DIR="$1"
T_VERSION="$2"
PYTHON_BIN="${PYTHON:-$(command -v python || true)}"
shift 2

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No active Python found; set PYTHON or activate a Python environment." >&2
  exit 127
fi
export BUNDLE PYTHON="$PYTHON_BIN"

DATASET_HAMMING="$("$PYTHON_BIN" - "$DATASET_DIR" <<'PY'
import pickle
import sys

with open(f"{sys.argv[1]}/params.pkl", "rb") as stream:
    params = pickle.load(stream)
if not isinstance(params, dict):
    params = vars(params)
print(params["min_hamming"])
PY
)"

exec "$PYTHON_BIN" "$VERDE_ROOT/train.py" \
  --reload_data "$DATASET_DIR" \
  --hamming "$DATASET_HAMMING" \
  --t_version "$T_VERSION" \
  "$@"
