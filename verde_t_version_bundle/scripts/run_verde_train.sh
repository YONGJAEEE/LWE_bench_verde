#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <verde-dataset-dir> <t-version> [additional train.py args ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERDE_ROOT="$(cd "$SCRIPT_DIR/../verde" && pwd)"
DATASET_DIR="$1"
T_VERSION="$2"
PYTHON_BIN="${PYTHON:-python}"
shift 2

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
