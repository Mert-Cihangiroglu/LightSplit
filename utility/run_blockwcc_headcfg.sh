#!/usr/bin/env bash
# D-7: does WCC worsen PCH under block_random (local) the way it does under dense fixed_random (-75%)?
# Utility: block_random b2 x {L2H_SCH,L1H_SCH,L2H_PCH,L1H_PCH} x wcc{0,0.01,0.1} x seeds{0,1},
#          + raw per head (baseline), CR8, IID (alpha=1e7), 100ep. cifar10, resnet18, 10 clients.
# Relative-to-raw feeds the head-config degradation figure (a_headcfg_block, WCC bars).
set -uo pipefail
cd "$(dirname "$0")"
PY=python
OUT=./reproduce/logs
mkdir -p results/csv logs

head_flags(){ case "$1" in
  L1H_SCH) echo "" ;; L1H_PCH) echo "--client-heads" ;;
  L2H_SCH) echo "--deep-head" ;; L2H_PCH) echo "--deep-head --client-heads" ;; esac; }

HEADS=(L2H_SCH L1H_SCH L2H_PCH L1H_PCH)
WCCS=(0 0.01 0.1)
SEEDS=(0 1)

# jobs: "block:head:wcc:seed"  (block '-' = raw)
JOBS=()
for S in "${SEEDS[@]}"; do for H in "${HEADS[@]}"; do
  JOBS+=("-:$H:0:$S")                                   # raw (wcc ignored)
  for W in "${WCCS[@]}"; do JOBS+=("2:$H:$W:$S"); done  # block_random b2 x wcc
done; done

run_one(){ # gpu block head wcc seed
  local g=$1 B=$2 H=$3 W=$4 S=$5 hf tag mode blockarg wccarg
  hf=$(head_flags "$H")
  if [ "$B" = "-" ]; then mode="raw"; blockarg=""; wccarg=""; tag="raw_${H}_s${S}";
  else mode="block_random"; blockarg="--block-size $B"; wccarg="--wcc-lambda $W"; tag="b${B}_${H}_w${W}_s${S}"; fi
  local sum="results/csv/blockwcc_${tag}_summary.csv"
  local log="$OUT/blockwcc_${tag}.log"
  echo "[$(date +%H:%M:%S)] GPU$g START $tag"
  $PY -u train_split.py --sweep --dataset cifar10 --modes $mode $blockarg $wccarg \
    --compression-ratio-list 8 --alpha-list 10000000 --num-clients 10 \
    --epochs 100 --patience 100 --seed "$S" --gpu "$g" $hf --no-viz --resume \
    --summary-csv "$sum" --epoch-csv "results/csv/blockwcc_${tag}_epochs.csv" > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$g DONE  $tag"
}

for g in 0 1 2 3; do
  ( i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r B H W S <<< "${JOBS[$i]}"
      run_one "$g" "$B" "$H" "$W" "$S"
      i=$((i+4))
    done ) &
done
wait
echo "[$(date +%H:%M:%S)] all block_random WCC x head-config utility runs complete"
