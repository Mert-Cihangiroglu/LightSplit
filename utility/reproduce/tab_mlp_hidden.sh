#!/usr/bin/env bash
# Table tab:mlp-hidden-ablation — LightSplit-L lift-back hidden width across the
# L1H/L2H x SCH/PCH cross at CR 8, IID, 10 clients (2000 epochs, patience 200).
# Widths: 256..2048 (CIFARs), 64..1024 (FashionMNIST, GTSRB); plus one Raw
# baseline per (dataset, config). 88 runs; a few GPU-days. SMOKE=1 runs two
# cells at 2 epochs. Raw cells that the paper reads from the full utility sweep
# are picked up automatically if tab_full_sweep.sh has run.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=mlp_hidden
JOBS="${JOBS:-1}"
DATASETS="${DATASETS:-cifar10 cifar100 fmnist gtsrb}"
HEADS="${HEADS:-L2H_SCH L1H_SCH L2H_PCH L1H_PCH}"
EP=2000; PAT=200
if [ "${SMOKE:-0}" = 1 ]; then DATASETS=cifar10; HEADS=L2H_SCH; EP=2; PAT=0; fi
mkdir -p "$OUT/$FIG"
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( run "$FIG" "$@" ) & }

widths() { case "$1" in cifar10|cifar100) echo "256 512 1024 2048" ;; *) echo "64 128 256 512 1024" ;; esac; }
[ "${SMOKE:-0}" = 1 ] && widths() { echo "256"; }

for ds in $DATASETS; do
  for H in $HEADS; do
    flags=(); [ "${H%%_*}" = L2H ] && flags+=(--deep-head); [ "${H##*_}" = PCH ] && flags+=(--client-heads)
    base=(--dataset "$ds" --num-clients 10 --alpha 10000000 --epochs "$EP")
    [ "$PAT" != 0 ] && base+=(--patience "$PAT")
    runq "${ds}_${H}_raw" "${base[@]}" --mode raw ${flags[@]+"${flags[@]}"}
    for h in $(widths "$ds"); do
      runq "${ds}_${H}_frll_h${h}" "${base[@]}" --mode fixed_random_ll --compression-ratio 8 \
        --mlp-hidden "$h" ${flags[@]+"${flags[@]}"}
    done
  done
done
wait
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_mlp_hidden.py" \
  --results "$OUT/$FIG" --fullsweep "$OUT/tab_full_sweep" --out "$OUT/tab_mlp_hidden.csv"
