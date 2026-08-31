#!/usr/bin/env bash
# Faithful narrow-cut (512-d) SDAR L7 USL privacy runs: client narrowed to width='cut8' (16,32,8 ->
# 512-d cut, standard feature body), SERVER kept standard-width via the 1x1 cut adapter (healthy victim,
# ~82% acc for the narrow cut), attacker decoder kept standard (full-strength). bottleneck none (narrowing IS the
# compressor). 3 seeds, compared against raw 0.572 in the compression cluster.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs
for run in 0 1 2; do
  gpu=$run
  log="$OUT/cut8_usl_l7_run${run}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START cut8 l7 run$run"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$run" \
    --width cut8 --server_width standard --attacker_width standard --print_to_stdout > "$log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] all cut8 runs complete"
echo "=== RESULTS (narrow cut 512-d, healthy victim) ==="
for run in 0 1 2; do
  s=$(grep "Average SSIM" "$OUT/cut8_usl_l7_run${run}.log" | tail -1)
  a=$(grep "Average label accuracy" "$OUT/cut8_usl_l7_run${run}.log" | tail -1)
  echo "run$run | ${s:-NO-SSIM} | ${a:-no-acc}"
done
