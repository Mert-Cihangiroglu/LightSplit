#!/usr/bin/env bash
# Multi-seed complement for the LightSplit-F privacy-vs-CR curve (SDAR L7 USL, cifar10, wcc=0).
# Single-run (run1) was contaminated by SDAR's seed-dependent victim collapse (e.g. cr4 victim dead
# at acc 0.003 the whole run -> SSIM meaningless). Fix = X4 convention: multiple seeds, then
# health-gate on victim task-acc and average the healthy seeds' SSIM.
# Here we add seeds 0 and 2 for every CR (run1 already exists) -> 3 seeds per CR. 22 runs, 4 GPUs.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

CRS=(4 8 16 32 64 128 256 512 1024 2048 4096)
SEEDS=(0 2)

# build the flat job list (cr,seed) and round-robin across 4 GPUs
JOBS=()
for cr in "${CRS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$cr:$s"); done; done

run_one() {
  local gpu="$1" cr="$2" seed="$3"
  local log="$OUT/fr_usl_l7_cr${cr}_run${seed}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START cr$cr seed$seed -> $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$seed" \
    --bottleneck_mode fixed_random --compression_ratio "$cr" --wcc_lambda 0 \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  cr$cr seed$seed"
}

# 4 per-GPU worker queues
for g in 0 1 2 3; do
  (
    i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r cr seed <<< "${JOBS[$i]}"
      run_one "$g" "$cr" "$seed"
      i=$((i+4))
    done
  ) &
done
wait
echo "[$(date +%H:%M:%S)] all seed runs complete"
echo "=== RAW RESULTS (cr, seed -> SSIM | victim acc) ==="
for cr in "${CRS[@]}"; do
  for seed in 0 1 2; do
    f="$OUT/fr_usl_l7_cr${cr}_run${seed}.log"
    [ -f "$f" ] || continue
    s=$(grep "Average SSIM" "$f" | tail -1 | grep -oE '[0-9.]+' | tail -1)
    a=$(grep "Average label accuracy" "$f" | tail -1 | grep -oE '[0-9.]+' | tail -1)
    echo "cr${cr} seed${seed}  ssim=${s:-NA}  acc=${a:-NA}"
  done
done
