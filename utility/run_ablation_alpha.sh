#!/usr/bin/env bash
# Alpha (Dirichlet concentration) ablation.
# Holds (num_clients=10, mal=0.1, poison=0.3, CR=16, seed=42, target=0, trigger=3).
# Skips alpha=10000000 since main sweep already covers it.
# 6 modes x 3 alpha values = 18 runs.
set -e

GPU="${GPU:-0}"
DATASET="${DATASET:-cifar10}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-results/backdoor}"
PYTHON="${PYTHON:-python}"

ALPHAS=(0.1 1.0 10.0)
MODES=(raw fixed_random fixed_random_ll fixed_random_conv learned_1x1)
CR=16
SEED=42

COMMON_ARGS=(
    --dataset "$DATASET"
    --epochs "$EPOCHS"
    --num-clients 10
    --malicious-ratio 0.1
    --poison-rate 0.3
    --target-class 0
    --trigger-size 3
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --out-dir "$OUT_DIR"
    --gpu "$GPU"
    --seed "$SEED"
)

for alpha in "${ALPHAS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "================================================================"
        echo "  alpha-ablation: alpha=$alpha mode=$mode"
        echo "================================================================"
        if [ "$mode" = "raw" ]; then
            "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --alpha "$alpha" --mode raw
        else
            "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --alpha "$alpha" --mode "$mode" --k "$CR"
        fi
    done
done

echo "Alpha ablation complete. Results in $OUT_DIR/"
