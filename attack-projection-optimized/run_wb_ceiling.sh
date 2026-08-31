#!/usr/bin/env bash
# White-box known-R reconstruction ceiling vs WCC lambda (decisive privacy run).
# lambda=0.1 first (the decisive cell), then lambda=0 (reproduces the ~0.6 baseline).
# CR8 first within each (highest/worst-case ceiling). Fresh 100-epoch victims so
# the lambda comparison is apples-to-apples. exp3 = known-R lift-and-decode.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
GPU="${GPU:-0}"
for L in 0.1 0; do
  echo "===================  WCC lambda=$L  ==================="
  "$PY" -u privileged_attack.py --mode fixed_random --deep-head \
    --cr-list 8 32 --experiments 3 --wcc-lambda "$L" \
    --split-epochs 100 --retrain --gpu "$GPU" \
    --output-dir "results/wb_ceiling_l$L" || echo "!! lambda=$L FAILED"
done
echo "===================  wb ceiling sweep complete  ==================="
