#!/usr/bin/env bash
# D-7 privacy side: SDAR SSIM for block_random b2 at WCC {0.01, 0.1} (wcc=0 already gated).
# b2 x wcc{0.01,0.1} x seeds{0..4}, USL L7 cifar10, CR8. 10 runs. Health-gate afterward.
# SELF-DEFERRING: waits for the utility harness utility WCC runs (train_split.py) to finish first,
# so the TF SDAR jobs don't contend with the PyTorch utility jobs. [t] bracket avoids self-match.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

echo "[$(date +%H:%M:%S)] waiting for the utility harness utility (train_split.py) runs to finish..."
while pgrep -f "[t]rain_split.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] utility done -> starting block_random WCC SDAR SSIM runs"

WCCS=(0.01 0.1)
SEEDS=(0 1 2 3 4)
JOBS=()
for w in "${WCCS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$w:$s"); done; done

run_one(){ local g=$1 w=$2 s=$3
  local log="$OUT/block_usl_l7_b2_cr8_wcc${w}_run${s}.log"
  echo "[$(date +%H:%M:%S)] GPU$g START b2 cr8 wcc$w seed$s"
  CUDA_VISIBLE_DEVICES=$g $PY run_sdar.py usl resnet cifar10 7 "$s" \
    --bottleneck_mode block_random --block_size 2 --compression_ratio 8 --wcc_lambda "$w" \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$g DONE  b2 cr8 wcc$w seed$s"
}
for g in 0 1 2 3; do
  ( i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r w s <<< "${JOBS[$i]}"
      run_one "$g" "$w" "$s"
      i=$((i+4))
    done ) &
done
wait
echo "[$(date +%H:%M:%S)] all block_random WCC SDAR SSIM runs complete"
