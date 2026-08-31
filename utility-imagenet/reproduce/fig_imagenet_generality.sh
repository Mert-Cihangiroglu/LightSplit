#!/usr/bin/env bash
# Figure fig:imagenet_generality -- ImageNet-1k top-1 as % of Raw at CR 8/16/32.
# Arms: raw; block_random b=2 (LightSplit-F, 2x2 tiles), top-k sparse, learned 1x1 and
# fixed_random (global R) at CR 8, 16, 32.  13 runs x 90 epochs, about one GPU-day each.
# Output: $OUT/fig_imagenet_generality.csv  (method, cr, best_top1, pct_of_raw)
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=fig_imagenet_generality
run raw --mode raw --compression-ratio 8 --epochs "$EPOCHS"
for cr in 8 16 32; do
  run "block_b2_cr$cr"     --mode block_random --block-size 2 --compression-ratio "$cr" --epochs "$EPOCHS"
  run "topk_sparse_cr$cr"  --mode topk_sparse                 --compression-ratio "$cr" --epochs "$EPOCHS"
  run "learned_1x1_cr$cr"  --mode learned_1x1                 --compression-ratio "$cr" --epochs "$EPOCHS"
  run "fixed_random_cr$cr" --mode fixed_random                --compression-ratio "$cr" --epochs "$EPOCHS"
done
"$PY" "$REPRO_DIR/aggregate_imagenet.py" --figure imagenet_generality --root "$OUT" --out "$OUT/fig_imagenet_generality.csv"
