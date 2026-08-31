#!/usr/bin/env bash
# D-7: locate the WCC threshold where per-client heads (PCH) collapse under block_random b2.
# The coarse sweep jumps 0 (-1%) -> 0.01 (-57%); this fills the gap to SEE the transition.
# block_random b2 x {L2H_PCH, L1H_PCH} x wcc{0.001,0.003,0.005,0.007} x seeds{0,1}, CR8, IID, 100ep.
# (SCH fine-WCC already exists in utillong_*; PCH is the collapse of interest.)
# SELF-DEFERRING behind the WHOLE current pipeline (train_split.py OR run_sdar.py) -> runs last, no
# contention/races. [t]/[r] brackets avoid pgrep self-match.
set -uo pipefail
cd "$(dirname "$0")"
PY=python
OUT=./reproduce/logs
mkdir -p results/csv logs

echo "[$(date +%H:%M:%S)] waiting for current utility + SDAR pipeline to finish..."
while pgrep -f "[t]rain_split.py" >/dev/null 2>&1 || pgrep -f "[r]un_sdar.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] clear -> starting fine PCH WCC sweep"

head_flags(){ case "$1" in L2H_PCH) echo "--deep-head --client-heads" ;; L1H_PCH) echo "--client-heads" ;; esac; }
HEADS=(L2H_PCH L1H_PCH)
WCCS=(0.001 0.003 0.005 0.007)
SEEDS=(0 1)
JOBS=(); for S in "${SEEDS[@]}"; do for H in "${HEADS[@]}"; do for W in "${WCCS[@]}"; do JOBS+=("$H:$W:$S"); done; done; done

run_one(){ local g=$1 H=$2 W=$3 S=$4 hf; hf=$(head_flags "$H")
  local tag="b2_${H}_w${W}_s${S}"; local log="$OUT/blockwccfine_${tag}.log"
  echo "[$(date +%H:%M:%S)] GPU$g START $tag"
  $PY -u train_split.py --sweep --dataset cifar10 --modes block_random --block-size 2 --wcc-lambda "$W" \
    --compression-ratio-list 8 --alpha-list 10000000 --num-clients 10 \
    --epochs 100 --patience 100 --seed "$S" --gpu "$g" $hf --no-viz --resume \
    --summary-csv "results/csv/blockwccfine_${tag}_summary.csv" \
    --epoch-csv "results/csv/blockwccfine_${tag}_epochs.csv" > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$g DONE  $tag"
}
for g in 0 1 2 3; do
  ( i=$g; while [ $i -lt ${#JOBS[@]} ]; do IFS=: read -r H W S <<< "${JOBS[$i]}"; run_one "$g" "$H" "$W" "$S"; i=$((i+4)); done ) &
done
wait
echo "[$(date +%H:%M:%S)] fine PCH WCC sweep complete"
echo "=== RESULTS (best_global_test_acc) ==="
for H in "${HEADS[@]}"; do for W in "${WCCS[@]}"; do for S in "${SEEDS[@]}"; do
  f="results/csv/blockwccfine_b2_${H}_w${W}_s${S}_summary.csv"; [ -f "$f" ] && echo "$H w$W s$S -> $(awk -F, 'NR==2{print $9}' "$f")"; done; done; done
