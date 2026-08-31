#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WCC coefficient SENSITIVITY sweep (real multi-client harness).
#
# Purpose: map test accuracy vs lambda_WCC at a fixed compression ratio, to
#   (1) check whether 32x + WCC truly collapses utility (the SDAR single-client
#       harness diverges to ~0%; this tests the real harness), and
#   (2) chart the safe lambda range (the WCC-coefficient-selection guide), and
#       whether WCC still helps at very large CR.
#
# CIFAR-10 IID (alpha->inf), 10 clients, LightSplit-F (fixed_random), L2H cut.
# Each lambda writes its OWN summary CSV (the --sweep resume key ignores
# wcc_lambda and deep_head, so a shared CSV would falsely skip cells).
#
# Usage:
#   GPU=1 CR=32 ./run_wcc_sensitivity.sh                 # high-CR curve
#   GPU=0 CR=16 ./run_wcc_sensitivity.sh                 # mid-CR curve
#   GPU=1 CR=32 HEAD=PCH ./run_wcc_sensitivity.sh        # per-client head
#   GPU=0 CR=8  LAMBDAS="0 0.01 0.1" ./run_wcc_sensitivity.sh
# ---------------------------------------------------------------------------
set -uo pipefail

GPU="${GPU:-0}"
CR="${CR:-32}"
HEAD="${HEAD:-SCH}"                       # SCH = shared head, PCH = per-client
EPOCHS="${EPOCHS:-100}"
LAMBDAS="${LAMBDAS:-0 0.001 0.003 0.01 0.03 0.1}"
ALPHA="${ALPHA:-10000000}"               # IID
DS=cifar10

case "$HEAD" in
  SCH) HEADFLAG="--deep-head" ;;
  PCH) HEADFLAG="--deep-head --client-heads" ;;
  *)   echo "HEAD must be SCH or PCH"; exit 2 ;;
esac

mkdir -p results/csv logs
echo "WCC sensitivity: $DS L2H-$HEAD cr=${CR} gpu=${GPU} lambdas=[${LAMBDAS}]"

for w in $LAMBDAS; do
  tag="wccsens_${DS}_L2H_${HEAD}_cr${CR}_w${w}"
  echo "[$(date +%H:%M:%S)] >>> ${tag}"
  python train_split.py --sweep --dataset "$DS" --epochs "$EPOCHS" \
    --num-clients 10 --modes fixed_random --compression-ratio-list "$CR" \
    --alpha-list "$ALPHA" $HEADFLAG --wcc-lambda "$w" --no-viz --gpu "$GPU" \
    --summary-csv "results/csv/${tag}.csv" \
    --epoch-csv "results/csv/${tag}_epochs.csv" \
    2>&1 | tee "logs/${tag}.log"
done
echo "[$(date +%H:%M:%S)] DONE  cr${CR} L2H-${HEAD}"
