#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
#
# Diagnostic experiments for ks=9 accuracy gap.
# 4 experiments, run sequentially on one GPU so you can watch output live.
#
# Usage:
#   bash run_diag_experiments.sh          # GPU 1 (default)
#   bash run_diag_experiments.sh --gpu 0  # GPU 0
#
set -euo pipefail

CONDA_ENV="LightSplit"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/train_split.py"

GPU=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="${REPO_ROOT}/results/fixed-random-conv-results"
SUMMARY="${RESULTS}/diag_v2_summary.csv"
EPOCH_CSV="${RESULTS}/diag_v2_epochs.csv"

# Common flags
COMMON="--mode fixed_random_conv --compression-ratio 8 --conv-kernel-size 9 \
  --num-clients 10 --alpha 1e7 --client-heads --epochs 100 \
  --batch-size 128 --gpu $GPU --seed 42 --no-viz \
  --summary-csv $SUMMARY --epoch-csv $EPOCH_CSV"

echo "═══════════════════════════════════════════════════════════════"
echo "  Diagnostic experiments — ks=9 accuracy gap"
echo "  GPU: $GPU | Results: $SUMMARY"
echo "  4 experiments, ~2-4 hrs each"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "━━━ [1/4] Constant LR (shallow decompress, no cosine) ━━━"
conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
  $COMMON --no-scheduler

echo ""
echo "━━━ [2/4] Larger adapter + cosine ━━━"
conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
  $COMMON --backbone-adapter

echo ""
echo "━━━ [3/4] Larger adapter + constant LR ━━━"
conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
  $COMMON --backbone-adapter --no-scheduler

echo ""
echo "━━━ [4/4] Detached bottleneck + larger adapter ━━━"
conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
  $COMMON --backbone-adapter --detach-bottleneck

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All 4 experiments done!"
echo "  Summary: $SUMMARY"
echo "  Epochs:  $EPOCH_CSV"
echo "═══════════════════════════════════════════════════════════════"
