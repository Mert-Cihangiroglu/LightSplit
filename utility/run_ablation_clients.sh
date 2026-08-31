#!/usr/bin/env bash
# Client-count ablation.
# Holds (alpha=1e7 IID, mal=0.1, poison=0.3, CR=16, seed=42, target=0, trigger=3).
# Skips num_clients=10 since main sweep already covers it.
# 6 modes x 3 client counts = 18 runs.
set -e

GPU="${GPU:-0}"
DATASET="${DATASET:-cifar10}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-results/backdoor}"
PYTHON="${PYTHON:-python}"

CLIENT_COUNTS=(5 20 50)
MODES=(raw fixed_random fixed_random_ll fixed_random_conv learned_1x1)
CR=16
SEED=42

COMMON_ARGS=(
    --dataset "$DATASET"
    --epochs "$EPOCHS"
    --alpha 10000000
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

for nc in "${CLIENT_COUNTS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "================================================================"
        echo "  clients-ablation: num_clients=$nc mode=$mode"
        echo "================================================================"
        if [ "$mode" = "raw" ]; then
            "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --num-clients "$nc" --mode raw
        else
            "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --num-clients "$nc" --mode "$mode" --k "$CR"
        fi
    done
done

echo "Clients ablation complete. Results in $OUT_DIR/"
