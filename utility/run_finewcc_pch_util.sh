#!/usr/bin/env bash
# D-7 fine WCC: UTILITY at PCH. Observe how finer WCC affects per-client-head accuracy under
# block_random b2 -- locate the collapse threshold in the 0->0.01 gap.
# b2 x {L2H_PCH, L1H_PCH} x wcc{0,0.001,0.003,0.005,0.007,0.01,0.03} x seeds{0,1}. CR8, IID, 100ep.
# Runs NOW (GPUs are free). Pairs with run_finewcc_ssim.sh (privacy at the same WCC).
set -uo pipefail
cd "$(dirname "$0")"
PY=python
OUT=./reproduce/logs
mkdir -p results/csv logs

head_flags(){ case "$1" in L2H_PCH) echo "--deep-head --client-heads" ;; L1H_PCH) echo "--client-heads" ;; esac; }
HEADS=(L2H_PCH L1H_PCH)
WCCS=(0 0.001 0.003 0.005 0.007 0.01 0.03)
SEEDS=(0 1)
JOBS=(); for S in "${SEEDS[@]}"; do for H in "${HEADS[@]}"; do for W in "${WCCS[@]}"; do JOBS+=("$H:$W:$S"); done; done; done

run_one(){ local g=$1 H=$2 W=$3 S=$4 hf; hf=$(head_flags "$H")
  local tag="b2_${H}_w${W}_s${S}"; local log="$OUT/finewccutil_${tag}.log"
  echo "[$(date +%H:%M:%S)] GPU$g START $tag"
  $PY -u train_split.py --sweep --dataset cifar10 --modes block_random --block-size 2 --wcc-lambda "$W" \
    --compression-ratio-list 8 --alpha-list 10000000 --num-clients 10 \
    --epochs 100 --patience 100 --seed "$S" --gpu "$g" $hf --no-viz --resume \
    --summary-csv "results/csv/finewccutil_${tag}_summary.csv" \
    --epoch-csv "results/csv/finewccutil_${tag}_epochs.csv" > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$g DONE  $tag"
}
for g in 0 1 2 3; do
  ( i=$g; while [ $i -lt ${#JOBS[@]} ]; do IFS=: read -r H W S <<< "${JOBS[$i]}"; run_one "$g" "$H" "$W" "$S"; i=$((i+4)); done ) &
done
wait
echo "[$(date +%H:%M:%S)] fine PCH utility sweep complete"
echo "=== RESULTS best_global_test_acc ==="
for H in "${HEADS[@]}"; do for W in "${WCCS[@]}"; do for S in "${SEEDS[@]}"; do
  f="results/csv/finewccutil_b2_${H}_w${W}_s${S}_summary.csv"; [ -f "$f" ] && echo "$H w$W s$S -> $(awk -F, 'NR==2{print $9}' "$f")"; done; done; done
