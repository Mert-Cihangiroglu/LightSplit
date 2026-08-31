#!/usr/bin/env bash
# CONTROLLED CHECK: does current-code WCC reproduce Table VII for fixed_random (LightSplit-F)?
# Table VII (LIGHTSPLIT-F L2H-SCH, CIFAR-10 IID, n=10, CR=8x): raw 87.24,
#   wcc 0 -> -5.0% | 0.001 -> -6.0% | 0.01 -> -7.5% | 0.1 -> -5.3%   (WCC barely hurts L2H-SCH)
# If our fixed_random instead craters at wcc=0.1, the current WCC is mis-scaled vs the paper.
# Writes to a FRESH results/wcc_check/ dir (does NOT touch existing results).
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY="${PY:-python}"
GPU="${GPU:-0}"
LAMS="${LAMS:-0 0.001 0.01 0.1}"
EPOCHS="${EPOCHS:-2000}"; PATIENCE="${PATIENCE:-100}"   # table recipe: train to convergence
mkdir -p results/wcc_check logs
for L in $LAMS; do
  echo "[$(date '+%F %T')] fixed_random L2H_SCH CR8 IID 10cl ${EPOCHS}ep(pat${PATIENCE}) wcc=$L (GPU $GPU)"
  PYTHONUNBUFFERED=1 $PY -u train_split.py --sweep \
    --modes fixed_random --compression-ratio-list 8 --alpha-list 10000000 \
    --num-clients 10 --deep-head --wcc-lambda "$L" \
    --dataset cifar10 --epochs "$EPOCHS" --patience "$PATIENCE" --gpu "$GPU" --resume --no-viz \
    --summary-csv "results/wcc_check/wcccheck_fr_L2H_SCH_w${L}_summary.csv" \
    --epoch-csv  "results/wcc_check/wcccheck_fr_L2H_SCH_w${L}_epochs.csv" \
    2>&1 | tee -a "logs/wcccheck_fr_w${L}.log"
done
echo "=== wcc_check done: LAMS={$LAMS} ==="
