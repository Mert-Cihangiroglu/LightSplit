#!/usr/bin/env bash
# Figure fig:imagenet_lsl_lsf -- LightSplit-L (lift-back width 4096) against LightSplit-F on
# ImageNet-1k at CR 8, four matched pairs:
#   adam30   Adam 1e-3, 30 epochs, full training set
#   adam100  Adam 1e-3, 100 epochs, full training set   (the paper's run stopped at epoch 89)
#   sgd90    SGD 0.1, 90 epochs, full training set       (LS-F arm shared with fig_imagenet_generality)
#   pc200    SGD 0.1, 577 epochs, 200 training images per class  (the paper's runs stopped at epoch 542/544)
# 8 runs; about one GPU-day per 90 full-data epochs.
# Output: $OUT/fig_imagenet_lsl_lsf.csv  (pair, lsl_best_top1, lsf_best_top1, ratio_best_pct, ratio_final_tenth_pct)
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=fig_imagenet_lsl_lsf
ep() { if [ "${FAKE:-0}" = 1 ]; then echo "$EPOCHS"; else echo "$1"; fi; }   # fixed schedules; EPOCHS only under FAKE=1
LSL=(--mode fixed_random_ll --ll-hidden 4096 --compression-ratio 8); LSF=(--mode fixed_random --compression-ratio 8)
run adam30_lsl  "${LSL[@]}" --optimizer adam --lr 1e-3 --epochs "$(ep 30)"
run adam30_lsf  "${LSF[@]}" --optimizer adam --lr 1e-3 --epochs "$(ep 30)"
run adam100_lsl "${LSL[@]}" --optimizer adam --lr 1e-3 --epochs "$(ep 100)"
run adam100_lsf "${LSF[@]}" --optimizer adam --lr 1e-3 --epochs "$(ep 100)"
run sgd90_lsl   "${LSL[@]}" --epochs "$EPOCHS"
if [ -f "$OUT/fig_imagenet_generality/fixed_random_cr8_epochs.csv" ]; then
  cp "$OUT/fig_imagenet_generality/fixed_random_cr8_epochs.csv" "$OUT/$FIG/sgd90_lsf_epochs.csv"
  cp "$OUT/fig_imagenet_generality/fixed_random_cr8_summary.csv" "$OUT/$FIG/sgd90_lsf_summary.csv"
else run sgd90_lsf "${LSF[@]}" --epochs "$EPOCHS"; fi
run pc200_lsl   "${LSL[@]}" --train-per-class 200 --epochs "$(ep 577)"
run pc200_lsf   "${LSF[@]}" --train-per-class 200 --epochs "$(ep 577)"
"$PY" "$REPRO_DIR/aggregate_imagenet.py" --figure imagenet_lsl_lsf --root "$OUT" --out "$OUT/fig_imagenet_lsl_lsf.csv"
