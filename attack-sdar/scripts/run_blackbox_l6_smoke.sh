#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

echo "=== SDAR level-6 smoke test: fixed_random CR=8 WCC=0, run=99, one iteration ==="
CUDA_VISIBLE_DEVICES="$GPU" \
  SETTING=vsl \
  MODEL=resnet \
  DATASET=cifar10 \
  LEVEL=6 \
  RUNS=99 \
  COMPRESSION_RATIO=8 \
  WCC_LAMBDAS=0 \
  NUM_ITER_OVERRIDE=1 \
  LOGDIR=blackbox_vsl_l6_smoke_logs \
  MPLCONFIGDIR=/tmp/mpl-sdar-vsl-l6-smoke \
  PYTHON_BIN="$PYTHON_BIN" \
  ./run_blackbox_fixedrandom_only.sh
