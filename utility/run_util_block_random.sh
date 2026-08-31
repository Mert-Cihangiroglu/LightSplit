#!/usr/bin/env bash
# ============================================================================
# block_random UTILITY grid (the utility harness deployment side), cifar10.
# Pairs with the SDAR block_random PRIVACY sweep. ONE head-combo lane per call
# so lanes can be staged across GPUs.
#   sweep (per lane): block_size in {1,2,4}  x  CR{8,16,32}  x  alpha{IID, 0.1}
#       block=1 per-position ... block=4 mid. block=8 dropped (==fixed_random).
#   fixed: cifar10, 4096-d cut (==SDAR L7), 10 clients, 100 ep, Adam 1e-3.
# Resume-safe (epoch-csv key includes block_size). One summary CSV per lane.
#
#   HEAD=L2H_SCH GPU=0 ./run_util_block_random.sh     # --deep-head
#   HEAD=L2H_PCH GPU=1 ./run_util_block_random.sh     # --deep-head --client-heads
#   HEAD=L1H_SCH GPU=2 ./run_util_block_random.sh     # (none)
#   HEAD=L1H_PCH GPU=3 ./run_util_block_random.sh     # --client-heads
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
GPU="${GPU:-0}"
HEAD="${HEAD:-L2H_SCH}"
DATASET="${DATASET:-cifar10}"
# per-dataset default block sizes (per-position .. largest non-global; the global
# block == fixed_random is dropped since it is already covered).
if [ -z "${BLOCKS:-}" ]; then
  case "$DATASET" in
    cifar10|cifar100) BLOCKS="1 2 4" ;;   # cut 8x8
    gtsrb)            BLOCKS="1 2 7" ;;    # cut 14x14
    fmnist)           BLOCKS="1 2 4" ;;    # cut 12x12
    *)                BLOCKS="1 2 4" ;;
  esac
fi
CRS="${CRS:-8 16 32}"
ALPHAS="${ALPHAS:-10000000 0.1}"
EPOCHS="${EPOCHS:-100}"
NUMCLIENTS="${NUMCLIENTS:-10}"

case "$HEAD" in
  L2H_SCH) FLAGS="--deep-head" ;;
  L2H_PCH) FLAGS="--deep-head --client-heads" ;;
  L1H_SCH) FLAGS="" ;;
  L1H_PCH) FLAGS="--client-heads" ;;
  *) echo "bad HEAD=$HEAD (want L2H_SCH|L2H_PCH|L1H_SCH|L1H_PCH)"; exit 1 ;;
esac

CSV="results/csv/util_block_random_${DATASET}_${HEAD}_summary.csv"
ECSV="results/csv/util_block_random_${DATASET}_${HEAD}_epochs.csv"
LOG="logs/util_block_random_${DATASET}_${HEAD}.log"
mkdir -p logs results/csv

echo "=== block_random utility lane ${HEAD} | ${DATASET} | GPU ${GPU} | blocks{${BLOCKS}} CR{${CRS}} alpha{${ALPHAS}} | ${EPOCHS}ep ==="
echo "flags='${FLAGS:-<none>}'  summary=${CSV}"; echo

for B in $BLOCKS; do
  echo "[$(date '+%F %T')] ${HEAD} block_size=${B} -> ${LOG}"
  PYTHONUNBUFFERED=1 $PY -u train_split.py --sweep \
    --modes block_random --block-size "$B" \
    --compression-ratio-list $CRS --alpha-list $ALPHAS \
    --num-clients "$NUMCLIENTS" $FLAGS \
    --dataset "$DATASET" --epochs "$EPOCHS" --gpu "$GPU" --resume --no-viz \
    --summary-csv "$CSV" --epoch-csv "$ECSV" \
    2>&1 | tee -a "$LOG"
done

echo "=== done lane ${HEAD} -> ${CSV} ==="
