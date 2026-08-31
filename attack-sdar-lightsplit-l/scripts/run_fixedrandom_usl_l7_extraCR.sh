#!/usr/bin/env bash
# Extend the LightSplit-F (fixed_random, wcc=0) SDAR L7 USL privacy-vs-CR curve to pair with the
# Task-accuracy-vs-CR utility curve, ACROSS THE FULL RANGE up to 4096x compression.
# Existing points (reused): CR8=0.360, CR16=0.167, CR32=0.084 (USL, run1, k=512/256/128).
# New points here: CR 4, 64, 128, 256, 512, 1024, 2048, 4096.
#   k = 4096/CR:  4->1024  64->64  128->32  256->16  512->8  1024->4  2048->2  4096->1  (clamped >=1).
# cifar10 / resnet / L7 / USL / run1 / num_iter default (20000) -- same budget as the existing curve.
# 8 runs pinned 2-per-GPU across GPUs 0-3, each GPU processing its queue sequentially.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs
RUN=1

run_one() {
  local gpu="$1" cr="$2"
  local log="$OUT/fr_usl_l7_cr${cr}_run${RUN}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START fixed_random USL l7 cr$cr run$RUN -> $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$RUN" \
    --bottleneck_mode fixed_random --compression_ratio "$cr" --wcc_lambda 0 \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  cr$cr"
}

# per-GPU sequential queues (2 CRs each), all GPUs in parallel
( run_one 0 4    ; run_one 0 512  ) &
( run_one 1 64   ; run_one 1 1024 ) &
( run_one 2 128  ; run_one 2 2048 ) &
( run_one 3 256  ; run_one 3 4096 ) &
wait

echo "[$(date +%H:%M:%S)] all fixed_random extra-CR runs complete"
echo "=== RESULTS (LightSplit-F wcc=0, USL L7, run$RUN) ==="
for cr in 4 64 128 256 512 1024 2048 4096; do
  s=$(grep "Average SSIM" "$OUT/fr_usl_l7_cr${cr}_run${RUN}.log" | tail -1)
  a=$(grep "Average label accuracy" "$OUT/fr_usl_l7_cr${cr}_run${RUN}.log" | tail -1)
  echo "cr$cr | ${s:-NO-SSIM} | ${a:-no-acc}"
done
