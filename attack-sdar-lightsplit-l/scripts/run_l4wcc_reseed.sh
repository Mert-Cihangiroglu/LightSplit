#!/usr/bin/env bash
# C-W5: firm up the L4 fixed_random WCC metric-divergence and RECOVER the collapsed cells.
# L4 USL cr8 = Table IV (non-monotonic SSIM, HEALTHY victim 0.87-0.97) -> multi-seed to confirm the
# bounce is robust. L4 cr32 wcc{0.01,0.1} COLLAPSED (acc 0.003/0.030) -> re-seed to see if a healthy
# victim is ever obtained (or if cr32+WCC fundamentally collapses).
# fixed_random x cr{8,32} x wcc{0,0.01,0.1} x seeds{0,2,3,4} (run1 already exists -> 5 total). 24 runs.
# Runs from THIS copy (fixed_random byte-identical to production; production untouched).
# SELF-DEFERRING: waits for the current utility (train_split.py) AND block WCC SDAR (run_sdar.py) to
# finish so nothing contends. [t]/[r] brackets avoid pgrep self-match.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

echo "[$(date +%H:%M:%S)] waiting for current utility + SDAR jobs to finish..."
while pgrep -f "[t]rain_split.py" >/dev/null 2>&1 || pgrep -f "[r]un_sdar.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] clear -> starting L4 WCC re-seed"

CRS=(8 32)
WCCS=(0 0.01 0.1)
SEEDS=(0 2 3 4)
JOBS=()
for cr in "${CRS[@]}"; do for w in "${WCCS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$cr:$w:$s"); done; done; done

run_one(){ local g=$1 cr=$2 w=$3 s=$4
  local log="$OUT/l4wcc_cr${cr}_wcc${w}_run${s}.log"
  echo "[$(date +%H:%M:%S)] GPU$g START L4 cr$cr wcc$w seed$s"
  CUDA_VISIBLE_DEVICES=$g $PY run_sdar.py usl resnet cifar10 4 "$s" \
    --bottleneck_mode fixed_random --compression_ratio "$cr" --wcc_lambda "$w" \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$g DONE  L4 cr$cr wcc$w seed$s"
}
for g in 0 1 2 3; do
  ( i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r cr w s <<< "${JOBS[$i]}"
      run_one "$g" "$cr" "$w" "$s"
      i=$((i+4))
    done ) &
done
wait
echo "[$(date +%H:%M:%S)] all L4 WCC re-seed runs complete"
echo "=== RESULTS (L4 fixed_random, cr, wcc, seed -> SSIM | victim acc) ==="
for cr in "${CRS[@]}"; do for w in "${WCCS[@]}"; do for s in "${SEEDS[@]}"; do
  f="$OUT/l4wcc_cr${cr}_wcc${w}_run${s}.log"; [ -f "$f" ] || continue
  ss=$(grep "Average SSIM" "$f"|tail -1|grep -oE '[0-9.]+'|tail -1)
  a=$(grep "Average label accuracy" "$f"|tail -1|grep -oE '[0-9.]+'|tail -1)
  echo "cr$cr wcc$w seed$s  ssim=${ss:-NA} acc=${a:-NA}"
done; done; done
