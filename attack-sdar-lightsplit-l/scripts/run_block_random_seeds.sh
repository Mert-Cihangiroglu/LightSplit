#!/usr/bin/env bash
# block_random SDAR privacy seed top-up (USL L7 cifar10, wcc=0) to eliminate victim collapse in the
# existing 2-seed block_usl_l7_ssim.csv. Adds seeds 2,3,4,5 for b1/b2/b4 x CR{8,16,32} -> 6 total
# seeds per cell (with existing 0,1). Health-gate on victim task-acc afterward, average healthy.
# Runs from THIS copy (block_random code is byte-identical to production; production untouched).
#
# SELF-DEFERRING: waits until the current fixed_random/fixed_random_ll top-up finishes (so we don't
# contend), then launches. The [f] bracket keeps pgrep from matching its own command line.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

echo "[$(date +%H:%M:%S)] waiting for current fixed_random/_ll runs to finish before starting..."
while pgrep -f "bottleneck_mode [f]ixed_random" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] GPUs free -> starting block_random seed top-up"

BLOCKS=(1 2 4)
CRS=(8 16 32)
SEEDS=(2 3 4 5)
JOBS=()
for b in "${BLOCKS[@]}"; do for cr in "${CRS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$b:$cr:$s"); done; done; done

run_one() {
  local gpu="$1" b="$2" cr="$3" seed="$4"
  local log="$OUT/block_usl_l7_b${b}_cr${cr}_run${seed}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START b$b cr$cr seed$seed -> $(basename $log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$seed" \
    --bottleneck_mode block_random --block_size "$b" --compression_ratio "$cr" --wcc_lambda 0 \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  b$b cr$cr seed$seed"
}

for g in 0 1 2 3; do
  (
    i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r b cr seed <<< "${JOBS[$i]}"
      run_one "$g" "$b" "$cr" "$seed"
      i=$((i+4))
    done
  ) &
done
wait
echo "[$(date +%H:%M:%S)] all block_random seed runs complete"
