#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LEVEL="${1:?Usage: $0 level gpu1|gpu2|gpu3}"
QUEUE="${2:?Usage: $0 level gpu1|gpu2|gpu3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_sweep() {
  local gpu="$1"
  local setting="$2"
  local cr="$3"
  local wcc_lambdas="$4"
  local logdir="blackbox_${setting}_l${LEVEL}_cr${cr}_logs"
  local mpldir="/tmp/mpl-sdar-${setting}-l${LEVEL}-fixed-random-cr${cr}-gpu${gpu}"

  echo
  echo "[$(date '+%F %T')] GPU ${gpu}: level=${LEVEL} ${setting} fixed_random CR=${cr} WCC={${wcc_lambdas}}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    SETTING="$setting" \
    MODEL=resnet \
    DATASET=cifar10 \
    LEVEL="$LEVEL" \
    RUNS=1 \
    COMPRESSION_RATIO="$cr" \
    WCC_LAMBDAS="$wcc_lambdas" \
    LOGDIR="$logdir" \
    MPLCONFIGDIR="$mpldir" \
    PYTHON_BIN="$PYTHON_BIN" \
    ./run_blackbox_fixedrandom_only.sh
}

echo "=== SDAR black-box fixed-random level-${LEVEL} queue: ${QUEUE} ==="
echo "python=${PYTHON_BIN}"

case "$QUEUE" in
  gpu1)
    run_sweep 1 vsl 8 "0 0.01 0.1"
    run_sweep 1 vsl 16 "0 0.01 0.1"
    ;;
  gpu2)
    run_sweep 2 usl 8 "0 0.01 0.1"
    run_sweep 2 usl 16 "0 0.01 0.1"
    ;;
  gpu3)
    run_sweep 3 vsl 32 "0 0.01 0.1"
    run_sweep 3 usl 32 "0 0.01 0.1"
    ;;
  *)
    echo "Unknown queue: ${QUEUE}" >&2
    exit 2
    ;;
esac

echo
echo "[$(date '+%F %T')] Queue ${QUEUE} complete."
