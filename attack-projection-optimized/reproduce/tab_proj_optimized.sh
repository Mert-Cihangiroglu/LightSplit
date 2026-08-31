#!/usr/bin/env bash
# Reproduce the projection-optimized passive-attacker table (SSIM by fused signals x CR).
#
#   bash reproduce/tab_proj_optimized.sh          # full run (several GPU-hours)
#   SMOKE=1 bash reproduce/tab_proj_optimized.sh  # tiny budgets, checks mechanics only
#
# Pass 1 trains the split model per CR (2000 epochs, early stop) and runs
# experiment 3 (z with and without R) and experiment 6 (z + gradient g).
# Pass 2 reuses pass 1's split checkpoints and runs experiment 6 with
# --fuse-pinv (z + g + zR^T), decoder-only. The aggregator then emits the
# table CSV. Both passes skip themselves if their summary.csv is complete.
set -uo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$OUT"
cd "$CODE_DIR"

EXTRA=()
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA=(--cr-list 8 --split-epochs 2 --attack-epochs 2 --max-train-samples 2048)
fi

rows() { [ -f "$1" ] && tail -n +2 "$1" | wc -l || echo 0; }

if [ "$(rows "$OUT/gradfusion/summary.csv")" -lt 2 ]; then
  "$PY" -u privileged_attack.py --dataset cifar10 --data-root "$DATA_ROOT" \
    --experiments 3 6 --cr-list 8 16 32 "${EXTRA[@]}" \
    --output-dir "$OUT/gradfusion"
else echo "[skip] gradfusion"; fi

if [ "$(rows "$OUT/gradfusion_pinv/summary.csv")" -lt 1 ]; then
  "$PY" -u privileged_attack.py --dataset cifar10 --data-root "$DATA_ROOT" \
    --experiments 6 --fuse-pinv --cr-list 8 16 32 "${EXTRA[@]}" \
    --load-split-checkpoint "$OUT/gradfusion/checkpoints" \
    --output-dir "$OUT/gradfusion_pinv"
else echo "[skip] gradfusion_pinv"; fi

"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_proj_optimized.py" \
  --summary "$OUT/gradfusion/summary.csv" \
  --summary-pinv "$OUT/gradfusion_pinv/summary.csv" \
  --out "$OUT/tab_proj_optimized.csv"
