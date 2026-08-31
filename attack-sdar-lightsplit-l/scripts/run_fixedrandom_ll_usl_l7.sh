#!/usr/bin/env bash
# LightSplit-L (learned MLP lift-back) privacy runs in the ISOLATED copy harness.
# fixed_random_ll = fixed R projection (client) + trained MLP lift-back (server), mlp_hidden=512.
# Minimal F-vs-L comparison: cifar10 / resnet / L7 / USL / wcc=0 / CR {8,16,32} / seeds {0,1,2}.
# Health-gate on victim task-acc (label_acc in log); average healthy seeds' SSIM, then compare to the
# LightSplit-F healthy numbers (CR8/16/32 = 0.360/0.167/0.084).
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

CRS=(8 16 32)
SEEDS=(0 1 2)
JOBS=()
for cr in "${CRS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$cr:$s"); done; done

run_one() {
  local gpu="$1" cr="$2" seed="$3"
  local log="$OUT/frll_usl_l7_cr${cr}_run${seed}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START LSL cr$cr seed$seed -> $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$seed" \
    --bottleneck_mode fixed_random_ll --mlp_hidden 512 --compression_ratio "$cr" --wcc_lambda 0 \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  cr$cr seed$seed"
}

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
echo "[$(date +%H:%M:%S)] all LightSplit-L runs complete"
echo "=== RAW RESULTS (LightSplit-L: cr, seed -> SSIM | victim acc) ==="
for cr in "${CRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    f="$OUT/frll_usl_l7_cr${cr}_run${seed}.log"
    [ -f "$f" ] || continue
    s=$(grep "Average SSIM" "$f" | tail -1 | grep -oE '[0-9.]+' | tail -1)
    a=$(grep "Average label accuracy" "$f" | tail -1 | grep -oE '[0-9.]+' | tail -1)
    echo "cr${cr} seed${seed}  ssim=${s:-NA}  acc=${a:-NA}"
  done
done
