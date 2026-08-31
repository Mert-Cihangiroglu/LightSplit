#!/usr/bin/env bash
# 6 bottleneck modes x 3 compression ratios x 3 seeds = 48 runs.
# raw is collapsed to a single CR (it has no compression) -> 3 runs total for raw.
#
# Usage:
#   bash run_backdoor_sweep.sh                      # full sweep, default args
#   GPU=0 EPOCHS=50 bash run_backdoor_sweep.sh      # override defaults via env
#
# Output goes to results/backdoor/{dataset}_{mode}_c{N}_mal{m}_pr{p}_k{k}_seed{s}/
set -e

GPU="${GPU:-0}"
DATASET="${DATASET:-cifar10}"
EPOCHS="${EPOCHS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
ALPHA="${ALPHA:-10000000}"
MAL_RATIO="${MAL_RATIO:-0.1}"
POISON_RATE="${POISON_RATE:-0.3}"
TARGET_CLASS="${TARGET_CLASS:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUT_DIR="${OUT_DIR:-results/backdoor}"

COMPRESSED_MODES=(fixed_random fixed_random_ll fixed_random_conv learned_1x1)
CRS=(8 16 32)
SEEDS=(42 123 2024)

COMMON_ARGS=(
    --dataset "$DATASET"
    --epochs "$EPOCHS"
    --num-clients "$NUM_CLIENTS"
    --alpha "$ALPHA"
    --malicious-ratio "$MAL_RATIO"
    --poison-rate "$POISON_RATE"
    --target-class "$TARGET_CLASS"
    --trigger-size "$TRIGGER_SIZE"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --out-dir "$OUT_DIR"
    --gpu "$GPU"
)

PYTHON="${PYTHON:-python}"

run_one() {
    local mode="$1" cr="$2" seed="$3"
    echo "================================================================"
    echo "  mode=$mode  cr=$cr  seed=$seed"
    echo "================================================================"
    if [ "$mode" = "raw" ]; then
        "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --mode raw --seed "$seed"
    else
        "$PYTHON" run_backdoor.py "${COMMON_ARGS[@]}" --mode "$mode" --k "$cr" --seed "$seed"
    fi
}

# raw: 3 runs (one per seed)
for seed in "${SEEDS[@]}"; do
    run_one raw 0 "$seed"
done

# Compressed modes: 5 modes x 3 CRs x 3 seeds = 45 runs
for mode in "${COMPRESSED_MODES[@]}"; do
    for cr in "${CRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$mode" "$cr" "$seed"
        done
    done
done

echo "Sweep complete. Results in $OUT_DIR/"
