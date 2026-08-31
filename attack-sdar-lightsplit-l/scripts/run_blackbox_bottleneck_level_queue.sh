#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BOTTLENECK_MODE="${1:?Usage: $0 learned_1x1 level gpu1|gpu2|gpu3}"
LEVEL="${2:?Usage: $0 learned_1x1 level gpu1|gpu2|gpu3}"
QUEUE="${3:?Usage: $0 learned_1x1 level gpu1|gpu2|gpu3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$BOTTLENECK_MODE" != "learned_1x1" ]]; then
  echo "Unsupported BOTTLENECK_MODE=${BOTTLENECK_MODE}; expected learned_1x1." >&2
  exit 2
fi

run_sweep() {
  local gpu="$1"
  local setting="$2"
  local cr="$3"
  local logdir="blackbox_${setting}_l${LEVEL}_${BOTTLENECK_MODE}_cr${cr}_logs"
  local mpldir="/tmp/mpl-sdar-${setting}-l${LEVEL}-${BOTTLENECK_MODE}-cr${cr}-gpu${gpu}"

  echo
  echo "[$(date '+%F %T')] GPU ${gpu}: level=${LEVEL} ${setting} ${BOTTLENECK_MODE} CR=${cr} WCC={0}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    SETTING="$setting" \
    MODEL=resnet \
    DATASET=cifar10 \
    LEVEL="$LEVEL" \
    RUNS=1 \
    BOTTLENECK_MODE="$BOTTLENECK_MODE" \
    COMPRESSION_RATIO="$cr" \
    WCC_LAMBDAS="0" \
    LOGDIR="$logdir" \
    MPLCONFIGDIR="$mpldir" \
    PYTHON_BIN="$PYTHON_BIN" \
    ./run_blackbox_bottleneck_only.sh
}

echo "=== SDAR black-box ${BOTTLENECK_MODE} level-${LEVEL} CR grid queue: ${QUEUE} ==="
echo "python=${PYTHON_BIN}"

case "$QUEUE" in
  gpu1)
    run_sweep 1 vsl 8
    run_sweep 1 vsl 16
    ;;
  gpu2)
    run_sweep 2 usl 8
    run_sweep 2 usl 16
    ;;
  gpu3)
    run_sweep 3 vsl 32
    run_sweep 3 usl 32
    ;;
  *)
    echo "Unknown queue: ${QUEUE}" >&2
    exit 2
    ;;
esac

echo
echo "[$(date '+%F %T')] Queue ${QUEUE} complete."
