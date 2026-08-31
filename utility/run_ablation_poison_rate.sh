#!/usr/bin/env bash
# Poison-rate ablation (fraction of malicious client's data that gets stamped).
# Holds (alpha=1e7 IID, num_clients=10, mal=0.1, CR=16, seed=42, target=0, trigger=3).
# Skips poison=0.3 since main sweep already covers it.
# 6 modes x 4 poison values = 24 runs.
set -e

GPU="${GPU:-0}"
DATASET="${DATASET:-cifar10}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-results/backdoor}"
PYTHON="${PYTHON:-python}"

POISON_RATES=(0.1 0.2 0.5 0.8)
MODES=(raw fixed_random fixed_random_ll fixed_random_conv learned_1x1)
CR=16
SEED=42

COMMON_ARGS=(
    --dataset "$DATASET"
    --epochs "$EPOCHS"
    --alpha 10000000
    --num-clients 10
    --malicious-ratio 0.1
    --target-class 0
    --trigger-size 3
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --out-dir "$OUT_DIR"
    --gpu "$GPU"
    --seed "$SEED"
)

for pr in "${POISON_RATES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "================================================================"
        echo "  poison-rate-ablation: poison_rate=$pr mode=$mode"
        echo "================================================================"
        if [ "$mode" = "raw" ]; then
            "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --poison-rate "$pr" --mode raw
        else
            "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --poison-rate "$pr" --mode "$mode" --k "$CR"
        fi
    done
done

echo "Poison-rate ablation complete. Results in $OUT_DIR/"
