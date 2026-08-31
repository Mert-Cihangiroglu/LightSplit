#!/usr/bin/env bash
# FSHA columns of tab:block_privacy_unsplit (block_privacy_attacks.tex): block-local projection under
# FSHA, b in {1,2,4} x CR in {8,16,32}, and b in {1,2,4} with DP (sigma 0.02) at CR 8; seeds 123/124/125,
# SSIM of 20 reconstructed images per run, mean over seeds.
# Cost: 3 seeds x 2 datasets x (10 + 3) attack cells x ~5 min on one GPU (~6.5 h).
# Output: $OUT/tab_block_privacy_fsha.csv
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$FSHA_DIR"
for ds in $DATASETS; do for s in $SEEDS; do
  done_if "$OUT/tab_block_privacy_fsha/${ds}_seed${s}_block/cr32/block_random/b4/reconstructions.npz" || \
  "$TF_PY" scripts/run_dpblock.py --which block --dataset "$ds" --seed "$s" --iterations "$ITERATIONS" --log-frequency "$LOG_FREQ" \
      --output-root "$OUT/tab_block_privacy_fsha" --run-name "${ds}_seed${s}_block"
  done_if "$OUT/tab_block_privacy_fsha/${ds}_seed${s}_blockdp/cr08/block_random/b4_dp/reconstructions.npz" || \
  "$TF_PY" scripts/run_dpblock.py --which blockdp --dataset "$ds" --seed "$s" --iterations "$ITERATIONS" --log-frequency "$LOG_FREQ" \
      --output-root "$OUT/tab_block_privacy_fsha" --run-name "${ds}_seed${s}_blockdp"
done; done
"$TORCH_PY" "$REPRO_DIR/aggregate_fsha.py" --table block_privacy_fsha --root "$OUT" --seeds $SEEDS \
    --datasets $DATASETS --out "$OUT/tab_block_privacy_fsha.csv"
