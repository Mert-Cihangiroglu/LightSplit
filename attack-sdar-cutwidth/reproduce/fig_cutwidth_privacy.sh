#!/usr/bin/env bash
# Figure fig:cutwidth, panel (a) -- SDAR USL reconstruction across the cut width d, CIFAR-10, level 7.
# The client head keeps its body and only the cut stage sets the channel count (--width cut8 ... cut128,
# d = 512 ... 8192); the server and attacker models stay at the standard width. Per width a Raw run and
# LightSplit-F at CR 8/16/32; 5 seeds. 100 runs of 20 000 iterations, roughly 1.3 GPU-hours each.
# Output: $OUT/fig_cutwidth_privacy.csv -- one row per (d, CR) with the strongest attack over the
# victim-healthy seeds (ssim_max, the plotted value), the median, and the health flags.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=fig_cutwidth_privacy
WIDTHS="${WIDTHS:-cut8 cut16 cut32 cut64 cut128}"
W=(--server_width standard --attacker_width standard --force_cut_adapter)
for s in $SEEDS; do for w in $WIDTHS; do
  sdar_run "raw_${w}_run${s}" usl resnet cifar10 7 "$s" --width "$w" "${W[@]}"
  for cr in 8 16 32; do
    sdar_run "fr_cr${cr}_wcc0_${w}_run${s}" usl resnet cifar10 7 "$s" --width "$w" "${W[@]}" \
        --bottleneck_mode fixed_random --compression_ratio "$cr"
  done
done; done
"$PY" "$CODE_DIR/scripts/aggregate_dsweep.py" --logdir "$OUT/$FIG/logs" --csv "$OUT/fig_cutwidth_privacy.csv"
