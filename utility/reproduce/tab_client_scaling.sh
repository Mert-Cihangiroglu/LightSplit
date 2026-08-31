#!/usr/bin/env bash
# Table tab:client_scaling_ablation — Raw / LightSplit-F / LightSplit-L at
# n in {25,50,100} clients (CR 8, L2H-SCH, IID; 2000 epochs, patience 200) on
# CIFAR-10/100; 18 runs, a few GPU-days. The n=10 column is read from the full
# utility sweep (run tab_full_sweep.sh first). SMOKE=1 runs one cell at 2 epochs.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=client_scaling
JOBS="${JOBS:-1}"
DATASETS="${DATASETS:-cifar10 cifar100}"
NS="${NS:-25 50 100}"
MODES="${MODES:-raw fixed_random fixed_random_ll}"
EP=2000; PAT=200
if [ "${SMOKE:-0}" = 1 ]; then DATASETS=cifar10; NS=25; MODES=raw; EP=2; PAT=0; fi
mkdir -p "$OUT/$FIG"
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( run "$FIG" "$@" ) & }

for ds in $DATASETS; do for n in $NS; do for mode in $MODES; do
  base=(--dataset "$ds" --mode "$mode" --num-clients "$n" --alpha 10000000 \
        --epochs "$EP" --deep-head)
  [ "$PAT" != 0 ] && base+=(--patience "$PAT")
  [ "$mode" != raw ] && base+=(--compression-ratio 8)
  runq "${ds}_n${n}_${mode}" "${base[@]}"
done; done; done
wait
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_client_scaling.py" \
  --results "$OUT/$FIG" --fullsweep "$OUT/tab_full_sweep" --out "$OUT/tab_client_scaling.csv"
