#!/usr/bin/env bash
# Lower compression ratios (1, 2, 4).
# Current state of the CR ladder (health-gated, victim acc >= 0.40):
#   CR4  -> 1/3 healthy (thin);  CR2 -> NEVER RUN;  CR16, CR32 -> 0/2 healthy (holes at the headline CRs).
# CR1 == no compression == the Raw baseline, already measured (L7 raw SSIM 0.572) -- not re-run here.
# This tops up so each low-CR point rests on >=3 healthy seeds:
#   CR2  x seeds 0,1,2   (missing entirely; k = 4096/2 = 2048)
#   CR4  x seeds 3,4,5,6 (top-up; k = 1024)
#   CR16 x seeds 3,4,5   (repair 0-healthy hole; k = 256)
#   CR32 x seeds 3,4,5   (repair 0-healthy hole; k = 128)
# fixed_random, wcc=0, USL, cifar10, resnet, L7 -- same protocol/budget as the existing ladder,
# so the new points are directly comparable to CR8=0.360 etc.
# SELF-DEFERRING behind the running fine-WCC SDAR job ([r]un_sdar.py) -- do NOT kill that job to
# start this one: killing it releases other self-deferring guards and cascades auto-starts.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

echo "[$(date +%H:%M:%S)] waiting for the running SDAR job to finish..."
while pgrep -f "[r]un_sdar.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] GPUs clear -> starting low-CR top-up"

# (CR:seed) job list
JOBS=()
for s in 0 1 2;     do JOBS+=("2:$s");  done
for s in 3 4 5 6;   do JOBS+=("4:$s");  done
for s in 3 4 5;     do JOBS+=("16:$s"); done
for s in 3 4 5;     do JOBS+=("32:$s"); done

run_one() {
  local gpu="$1" cr="$2" seed="$3"
  local log="$OUT/fr_usl_l7_cr${cr}_run${seed}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START cr$cr seed$seed -> $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$seed" \
    --bottleneck_mode fixed_random --compression_ratio "$cr" --wcc_lambda 0 \
    --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  cr$cr seed$seed"
}

for g in 0 1 2 3; do
  ( i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r cr seed <<< "${JOBS[$i]}"
      run_one "$g" "$cr" "$seed"
      i=$((i+4))
    done ) &
done
wait

echo "[$(date +%H:%M:%S)] low-CR top-up complete"
echo "=== RESULTS (fixed_random wcc=0, USL L7) : SSIM | victim acc | gate ==="
for cr in 2 4 16 32; do
  for s in 0 1 2 3 4 5 6; do
    f="$OUT/fr_usl_l7_cr${cr}_run${s}.log"; [ -f "$f" ] || continue
    ssim=$(grep -oE "Average SSIM over all client's images: [0-9.]+" "$f" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    acc=$(grep -oE "Average label accuracy over all client's images: [0-9.]+" "$f" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    [ -z "$ssim" ] && { echo "cr$cr seed$s | unfinished"; continue; }
    g=$(awk -v a="$acc" 'BEGIN{print (a>=0.40)?"HEALTHY":"collapsed"}')
    echo "cr$cr seed$s | SSIM=$ssim | acc=$acc | $g"
  done
done
