#!/usr/bin/env bash
# ============================================================================
# block_random UTILITY x WCC sweep (the utility harness deployment side), cifar10.
# PAIRS with the privileged block_random PRIVACY (SSIM) sweep: SAME block dial
# x SAME WCC ladder, so utility-at-lambda lines up with SSIM-at-lambda.
#   grid (per call): block in BLOCKS  x  CR in CRS  x  wcc in WCCS, alpha=IID
#   fixed: cifar10, 10 clients, 100 ep, head=L2H_SCH (--deep-head).
# Per-(wcc) CSV because train_split's resume key omits wcc_lambda (same as the
# existing wccsens_* files) -> different lambdas must not share a CSV.
# Resume-safe. Split across GPUs by passing disjoint WCCS:
#   WCCS="0 0.1 0.3" GPU=0 ./run_util_block_random_wcc.sh    # lane A
#   WCCS="1 3 10"    GPU=1 ./run_util_block_random_wcc.sh    # lane B
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"   # NOTE: LightSplit env does not exist here
GPU="${GPU:-0}"
HEAD="${HEAD:-L2H_SCH}"
DATASET="${DATASET:-cifar10}"
BLOCKS="${BLOCKS:-1 2 4 8}"            # full dial (b8 == fixed_random endpoint, kept to match SSIM sweep)
CRS="${CRS:-8 32}"                     # match the SSIM sweep's --cr-list 8 32
ALPHAS="${ALPHAS:-10000000}"          # IID (matches the single-pool privileged victim)
WCCS="${WCCS:-0 0.1 0.3 1 3 10}"      # match the SSIM sweep's WCC ladder
EPOCHS="${EPOCHS:-100}"
NUMCLIENTS="${NUMCLIENTS:-10}"

case "$HEAD" in
  L2H_SCH) FLAGS="--deep-head" ;;
  L2H_PCH) FLAGS="--deep-head --client-heads" ;;
  L1H_SCH) FLAGS="" ;;
  L1H_PCH) FLAGS="--client-heads" ;;
  *) echo "bad HEAD=$HEAD (want L2H_SCH|L2H_PCH|L1H_SCH|L1H_PCH)"; exit 1 ;;
esac
mkdir -p logs results/csv
echo "=== block_random UTILITYxWCC | ${HEAD} ${DATASET} GPU${GPU} | blocks{${BLOCKS}} CR{${CRS}} wcc{${WCCS}} alpha{${ALPHAS}} ${EPOCHS}ep ==="

for L in $WCCS; do
  CSV="results/csv/utilwcc_block_random_${DATASET}_${HEAD}_w${L}_summary.csv"
  ECSV="results/csv/utilwcc_block_random_${DATASET}_${HEAD}_w${L}_epochs.csv"
  for B in $BLOCKS; do
    echo "[$(date '+%F %T')] wcc=${L} block=${B} -> ${CSV}"
    PYTHONUNBUFFERED=1 $PY -u train_split.py --sweep \
      --modes block_random --block-size "$B" \
      --compression-ratio-list $CRS --alpha-list $ALPHAS \
      --num-clients "$NUMCLIENTS" $FLAGS \
      --wcc-lambda "$L" \
      --dataset "$DATASET" --epochs "$EPOCHS" --gpu "$GPU" --resume --no-viz \
      --summary-csv "$CSV" --epoch-csv "$ECSV" \
      2>&1 | tee -a "logs/utilwcc_block_random_${DATASET}_${HEAD}_w${L}.log"
  done
done
echo "=== done lane WCCS={${WCCS}} HEAD=${HEAD} ==="
