#!/usr/bin/env bash
# Table tab:fsha_active -- FSHA fully-malicious attack at CR 8x, MNIST and Fashion-MNIST.
# Rows: Raw, Raw+DP, Learned 1x1, top-K sparse, LS-F (wcc 0 / 0.1 / 0.3), LS-F+DP.
# Attack cells: seeds 123 124 125 (20 reconstructed images each). Accuracy: seed 123.
# Cost: 48 attack cells x ~5 min + 16 clean cells x ~1.5 min on one GPU (~4.5 h).
# Output: $OUT/tab_fsha_active.csv  (one row per table row; mean and std over seeds).
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$FSHA_DIR"
for ds in $DATASETS; do
  for s in $SEEDS; do
    run="${ds}_seed${s}"
    done_if "$OUT/tab_fsha_active/$run/cr08/learned_1x1/wcc0/reconstructions.npz" || \
    "$TF_PY" scripts/run_bottleneck_grid.py --dataset "$ds" --seed "$s" --cr 8 --wcc 0 0.1 0.3 \
        --modes fixed_random topk_sparse learned_1x1 "${COMMON[@]}" \
        --output-root "$OUT/tab_fsha_active" --run-name "$run"
    done_if "$OUT/tab_fsha_active/${run}_dp/cr08/fixed_random/wcc0_dp/reconstructions.npz" || \
    "$TF_PY" scripts/run_dpblock.py --which dp --dataset "$ds" --seed "$s" --iterations "$ITERATIONS" --log-frequency "$LOG_FREQ" \
        --output-root "$OUT/tab_fsha_active" --run-name "${run}_dp"
  done
  cds="${ds}_seed123"
  done_if "$OUT/tab_fsha_active_clean/$cds/cr08/learned_1x1/wcc0/metrics.json" || \
  "$TF_PY" scripts/run_clean_accuracy_grid.py --dataset "$ds" --seed 123 --cr 8 --wcc 0 0.1 0.3 \
      --modes fixed_random topk_sparse learned_1x1 --epochs "$CLEAN_EPOCHS" --keep-going \
      --output-root "$OUT/tab_fsha_active_clean" --run-name "$cds"
  done_if "$OUT/tab_fsha_active_clean/${cds}_dp/cr08/fixed_random/wcc0_dp/metrics.json" || \
  "$TF_PY" scripts/run_clean_dpblock.py --which dp --dataset "$ds" --seed 123 --epochs "$CLEAN_EPOCHS" \
      --output-root "$OUT/tab_fsha_active_clean" --run-name "${cds}_dp"
done
"$TORCH_PY" "$REPRO_DIR/aggregate_fsha.py" --table fsha_active --root "$OUT" \
    --seeds $SEEDS --datasets $DATASETS --out "$OUT/tab_fsha_active.csv"
