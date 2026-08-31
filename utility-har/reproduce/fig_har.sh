#!/usr/bin/env bash
# Figures fig:har_cr8 (main text), fig:har_generality and fig:har_privacy (appendix) -- UCI HAR, 256-d MLP cut.
# Rows: Raw and, at CR 8/16/32 (k = 32/16/8): LightSplit-F, LightSplit-F+WCC (lambda 0.1), LightSplit-L,
# LightSplit-L+WCC, learned 1x1 ("codec"), top-K sparse; 3 seeds; per cell test accuracy, median
# per-feature reconstruction R^2 and subject re-identification accuracy.
# 57 cells x (120-epoch training + 3 attacks), a few minutes each on CPU.
# Output: $OUT/fig_har.csv
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=fig_har
cd "$CODE_DIR"
for s in $SEEDS; do cell none 0 none "$s"; done
for m in fixed:0 fixed:0.1 learned:0 learned:0.1 codec:0 topk_sparse:0; do
  for k in 32 16 8; do for s in $SEEDS; do cell "${m%%:*}" "${m##*:}" "$k" "$s"; done; done
done
"$PY" "$REPRO_DIR/aggregate_har.py" --root "$OUT" --seeds $SEEDS --out "$OUT/fig_har.csv"
