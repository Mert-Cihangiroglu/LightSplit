#!/usr/bin/env bash
# Figure fig:cutwidth, panel (b) -- utility across the cut width d on CIFAR-10.
# The deep head keeps its native-width body (--head-fixed-body) and only the cut stage sets the
# channel count C in {8,16,32,64,128}, i.e. d = 64*C in {512,...,8192}; per width a Raw run and
# LightSplit-F at CR 8/16/32; single client, 100 epochs, 3 seeds. 60 runs, ~15 min each on one GPU.
# Output: $OUT/fig_cutwidth_utility.csv  (method, cut_channels, d, cr, test_acc, test_acc_std, n_seeds)
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=fig_cutwidth_utility
SEEDS="${SEEDS:-42 43 123}"; EPOCHS="${EPOCHS:-100}"; CUTS="${CUTS:-8 16 32 64 128}"
for C in $CUTS; do for s in $SEEDS; do
  base=(--dataset cifar10 --cut-channels "$C" --head-fixed-body --deep-head --num-clients 1 --alpha 1e7 --epochs "$EPOCHS" --seed "$s")
  run "$FIG" "cut${C}_raw_seed${s}" "${base[@]}" --mode raw
  for cr in 8 16 32; do run "$FIG" "cut${C}_cr${cr}_seed${s}" "${base[@]}" --mode fixed_random --compression-ratio "$cr"; done
done; done
"$PY" "$REPRO_DIR/aggregate_cutwidth.py" --root "$OUT" --seeds $SEEDS --cuts $CUTS --out "$OUT/fig_cutwidth_utility.csv"
