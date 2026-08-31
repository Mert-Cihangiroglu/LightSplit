#!/usr/bin/bash
# ============================================================================
#  Kernel-size ablation sweep for fixed_random_conv bottleneck
#
#  Axes:
#    Dataset:      cifar10, cifar100
#    Alpha:        0.1 (non-IID), 1e7 (IID)
#    Kernel size:  1, 3, 5, 7, 9
#    Client heads: shared (no flag) vs per-client (--client-heads)
#
#  Fixed:
#    Backbone: ResNet18 (no --attn-layer1, no --backbone vit)
#    CR=8, 10 clients, 100 epochs, Adam lr=1e-3, cosine annealing
#
#  Parallelizes across 4 GPUs (one job per GPU at a time).
#  Each (dataset, alpha) combo gets its own GPU.
#
#  Results saved to:
#    results/fixed-random-conv-results/ks_ablation_summary.csv
#    results/fixed-random-conv-results/ks_ablation_epochs.csv
#
#  Supports --resume to skip already-completed runs.
#  Usage:
#    bash run_ks_ablation.sh
# ============================================================================
set -eu

PYTHON="python"
SCRIPT="train_split.py"
LOGDIR="/tmp/ks_ablation_logs"
mkdir -p "$LOGDIR"

RESULTS_DIR="results/ks-ablation-resnet18"
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/ks_ablation_summary.csv"
EPOCH_CSV="$RESULTS_DIR/ks_ablation_epochs.csv"

EPOCHS=100
NUM_CLIENTS=10
CR=8
LR=0.001
BATCH_SIZE=128

KERNEL_SIZES="1 3 5 7 9"
CLIENT_HEADS_OPTS="shared client_heads"

# ── Worker function: runs all KS x client_heads for one (dataset, alpha) ──
run_worker() {
    local DATASET=$1
    local ALPHA=$2
    local GPU=$3

    if [ "$ALPHA" = "10000000" ]; then
        ALPHA_TAG="iid"
    else
        ALPHA_TAG="a${ALPHA}"
    fi

    local LOGFILE="$LOGDIR/${DATASET}_${ALPHA_TAG}_gpu${GPU}.log"
    echo "[GPU $GPU] Starting: $DATASET alpha=$ALPHA_TAG → $LOGFILE"

    for KS in $KERNEL_SIZES; do
        for CH_MODE in $CLIENT_HEADS_OPTS; do
            CH_FLAG=""
            CH_LABEL="shared"
            if [ "$CH_MODE" = "client_heads" ]; then
                CH_FLAG="--client-heads"
                CH_LABEL="per-client"
            fi

            echo "[GPU $GPU] $DATASET | $ALPHA_TAG | ks=$KS | heads=$CH_LABEL" >> "$LOGFILE"

            $PYTHON $SCRIPT \
                --dataset "$DATASET" \
                --mode fixed_random_conv \
                --compression-ratio $CR \
                --conv-kernel-size "$KS" \
                --num-clients $NUM_CLIENTS \
                --alpha "$ALPHA" \
                --epochs $EPOCHS \
                --batch-size $BATCH_SIZE \
                --lr $LR \
                --gpu "$GPU" \
                --no-viz \
                --resume \
                $CH_FLAG \
                --summary-csv "$SUMMARY_CSV" \
                --epoch-csv "$EPOCH_CSV" \
                >> "$LOGFILE" 2>&1

        done
    done

    echo "[GPU $GPU] Done: $DATASET alpha=$ALPHA_TAG"
}

echo "============================================================"
echo "  Kernel-Size Ablation Sweep (4 GPUs parallel)"
echo "  40 experiments: 2 datasets × 2 alphas × 5 KS × 2 heads"
echo "  10 experiments per GPU"
echo "  Results: $RESULTS_DIR/"
echo "  Logs:    $LOGDIR/"
echo "============================================================"

# GPU 0 is reserved (other experiments running).
# Use GPUs 1-3: GPU 1 gets 2 jobs sequentially, GPUs 2-3 get 1 each.
(run_worker cifar10  10000000 1 && run_worker cifar10  0.1 1) &
run_worker cifar100 10000000 2 &
run_worker cifar100 0.1      3 &

echo ""
echo "3 GPU workers launched (GPU 0 reserved):"
echo "  GPU 1: cifar10 IID → cifar10 non-IID (20 exps, sequential)"
echo "  GPU 2: cifar100 IID (10 exps)"
echo "  GPU 3: cifar100 non-IID (10 exps)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGDIR/cifar10_iid_gpu1.log"
echo "  tail -f $LOGDIR/cifar10_a0.1_gpu1.log"
echo "  tail -f $LOGDIR/cifar100_iid_gpu2.log"
echo "  tail -f $LOGDIR/cifar100_a0.1_gpu3.log"
echo ""

# Wait for all workers
wait

echo ""
echo "============================================================"
echo "  All 40 experiments complete!"
echo "  Summary: $SUMMARY_CSV"
echo "  Epochs:  $EPOCH_CSV"
echo "============================================================"
