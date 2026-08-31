#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DP utility sweep on the CANONICAL the utility harness harness (baselines must live here,
# not an external tree). DP = per-sample L2 clip + Gaussian noise on the transmitted
# payload (std = sigma * clip), applied train+eval. Grid: CR x clip x sigma x {L1H,L2H}.
# CIFAR-10 IID, 10 clients, shared head. clip_norm=0 -> no DP (== plain baseline).
#
#   MODE=raw           -> DP-on-raw baseline (no compression; full activations)
#   MODE=fixed_random  -> LightSplit-F + DP (our method + DP)
#
# Resume-guarded: a cell whose summary CSV exists is skipped. Detach in tmux/setsid.
#
# Usage:
#   GPU=0 MODE=raw          SIGMAS="0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5" ./run_dp_util.sh
#   GPU=1 MODE=fixed_random CRS="8 16 32" SIGMAS="0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5" ./run_dp_util.sh
#   GPU=0 MODE=raw CRS=1 CLIPS=10.0 SIGMAS=0.5 EPOCHS=2 ./run_dp_util.sh   # smoke
# ---------------------------------------------------------------------------
set -uo pipefail
PY="${PY:-python}"
GPU="${GPU:-0}"
MODE="${MODE:-raw}"
SIGMAS="${SIGMAS:-0.0 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.4 0.5}"
CLIPS="${CLIPS:-10.0}"
EPOCHS="${EPOCHS:-100}"
if [ "$MODE" = "raw" ]; then CRS="${CRS:-1}"; else CRS="${CRS:-8 16 32}"; fi

mkdir -p results/csv logs
echo "DP util (the utility harness): mode=${MODE} gpu=${GPU} CRs=[${CRS}] clips=[${CLIPS}] sigmas=[${SIGMAS}]"

for CR in $CRS; do
  for CLIP in $CLIPS; do
    for SIG in $SIGMAS; do
      for DEEPTAG in L1H L2H; do
        DEEP=""; [ "$DEEPTAG" = "L2H" ] && DEEP="--deep-head"
        CRARG=""; [ "$MODE" != "raw" ] && CRARG="--compression-ratio $CR"
        tag="dpfull_${MODE}_${DEEPTAG}_SCH_cr${CR}_clip${CLIP}_sig${SIG}"
        if [ -s "results/csv/${tag}.csv" ]; then
          echo "[$(date +%H:%M:%S)] skip ${tag}"; continue
        fi
        echo "[$(date +%H:%M:%S)] >>> ${tag}"
        $PY train_split.py --dataset cifar10 --epochs "$EPOCHS" --num-clients 10 \
          --alpha 10000000 --mode "$MODE" $CRARG --clip-norm "$CLIP" --noise-multiplier "$SIG" \
          $DEEP --no-viz --gpu "$GPU" \
          --summary-csv "results/csv/${tag}.csv" \
          --epoch-csv "results/csv/${tag}_epochs.csv" \
          2>&1 | tee "logs/${tag}.log"
      done
    done
  done
done
echo "[$(date +%H:%M:%S)] DP util done: mode=${MODE}"
