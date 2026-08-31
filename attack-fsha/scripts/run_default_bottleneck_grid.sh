#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x python ]]; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_bottleneck_grid.py" \
  --dataset mnist \
  --output-root "${REPO_DIR}/results/bottleneck_grid" \
  --iterations 10000 \
  --log-frequency 500 \
  --batch-size 64 \
  --id-setup 3 \
  --cr 8 16 32 \
  --wcc 0 0.01 0.1 \
  "$@"
