#!/usr/bin/env bash
# Table tab:block_util_F/L (merged block-local utility grid).
# Grid: dataset x {IID, non-IID} x {L1H,L2H}x{SCH,PCH} x CR {8,16,32}
#       x block size b {1,2,4} x lift-back {transpose (F), learned MLP (L)}.
# One --sweep process per (dataset, head config, b) emits 12 rows
# (2 modes x 3 CR x 2 distributions); 44 processes, 528 runs total at
# 2000 epochs / patience 100 (several GPU-weeks; use JOBS to parallelise).
# GTSRB skips b=4 (the tile does not divide the activation).
# The Raw reference cells come from the full utility sweep: run
# tab_full_sweep.sh first (its Raw rows are enough).
# SMOKE=1 runs one cell at 2 epochs.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=blockutil
JOBS="${JOBS:-1}"
DATASETS="${DATASETS:-cifar10 cifar100 fmnist gtsrb}"
HEADS="${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH}"
BLOCKS="${BLOCKS:-1 2 4}"
CRS="${CRS:-8 16 32}"
EP=2000; PAT=100; EXPECT=12
if [ "${SMOKE:-0}" = 1 ]; then
  DATASETS=cifar10; HEADS=L2H_SCH; BLOCKS=2; CRS=8; EP=2; PAT=0; EXPECT=2
fi
mkdir -p "$OUT/$FIG"
rows() { [ -f "$1" ] && tail -n +2 "$1" | grep -c . || echo 0; }
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( "$@" ) & }

for ds in $DATASETS; do
  case "$ds" in cifar100) NA=0.5 ;; *) NA=0.1 ;; esac
  ALPHAS="10000000 $NA"; [ "${SMOKE:-0}" = 1 ] && ALPHAS="10000000"
  for H in $HEADS; do
    flags=(); [ "${H%%_*}" = L2H ] && flags+=(--deep-head); [ "${H##*_}" = PCH ] && flags+=(--client-heads)
    for B in $BLOCKS; do
      [ "$ds" = gtsrb ] && [ "$B" = 4 ] && continue
      sum="$OUT/$FIG/util2k_${ds}_${H}_b${B}_summary.csv"
      if [ "$(rows "$sum")" -ge "$EXPECT" ]; then echo "[skip] $ds $H b$B"; continue; fi
      echo "[$(date +%H:%M:%S)] $ds $H b$B"
      runq "$PY" "$CODE_DIR/train_split.py" --sweep --dataset "$ds" --data-root "$DATA_ROOT" \
        --modes block_random_ll block_random --block-size "$B" \
        --compression-ratio-list $CRS --alpha-list $ALPHAS --num-clients 10 \
        --epochs "$EP" --patience "$PAT" ${flags[@]+"${flags[@]}"} --gpu 0 --no-viz \
        --summary-csv "$sum" --epoch-csv "$OUT/$FIG/util2k_${ds}_${H}_b${B}_epochs.csv"
    done
  done
done
wait
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_block_dp.py" \
  --util2k "$OUT/$FIG" --fullsweep "$OUT/tab_full_sweep" \
  --out-block "$OUT/tab_block_util_FL.csv"
