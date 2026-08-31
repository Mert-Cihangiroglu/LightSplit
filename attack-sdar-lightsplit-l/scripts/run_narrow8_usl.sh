#!/usr/bin/env bash
# Narrow-cut (512-d, CR8-matched) SDAR L7 USL privacy runs for the baselines box.
# Client model narrowed to width='narrow8' (8x8x8 = 512-d cut, no projection: bottleneck none);
# the reconstruction attacker stays at standard width (--attacker_width standard) so the narrow
# cut faces the SAME full-strength decoder as the raw / fixed_random / top-k bars. 3 seeds.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs
mkdir -p "$OUT"

for run in 0 1 2; do
  gpu=$run
  log="$OUT/narrow8_usl_l7_run${run}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START narrow8 l7 run$run -> $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$run" \
    --width narrow8 --attacker_width standard --print_to_stdout > "$log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] all narrow8 runs complete"
echo "=== SSIM results ==="
for run in 0 1 2; do
  s=$(grep "Average SSIM" "$OUT/narrow8_usl_l7_run${run}.log" | tail -1)
  a=$(grep "Average label accuracy" "$OUT/narrow8_usl_l7_run${run}.log" | tail -1)
  echo "run$run | ${s:-NO-SSIM} | ${a:-no-acc}"
done
