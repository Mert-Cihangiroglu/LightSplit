#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BOTTLENECK_MODE="${1:?Usage: $0 learned_1x1 gpu1|gpu2|gpu3}"
QUEUE="${2:?Usage: $0 learned_1x1 gpu1|gpu2|gpu3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$BOTTLENECK_MODE" != "learned_1x1" ]]; then
  echo "Unsupported BOTTLENECK_MODE=${BOTTLENECK_MODE}; expected learned_1x1." >&2
  exit 2
fi

run_sweep() {
  local gpu="$1"
  local setting="$2"
  local cr="$3"
  local wcc_lambdas="$4"
  local logdir="$5"
  local mpldir="$6"

  echo
  echo "[$(date '+%F %T')] GPU ${gpu}: ${setting} ${BOTTLENECK_MODE} CR=${cr} WCC={${wcc_lambdas}}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    SETTING="$setting" \
    MODEL=resnet \
    DATASET=cifar10 \
    LEVEL=7 \
    RUNS=1 \
    BOTTLENECK_MODE="$BOTTLENECK_MODE" \
    COMPRESSION_RATIO="$cr" \
    WCC_LAMBDAS="$wcc_lambdas" \
    LOGDIR="$logdir" \
    MPLCONFIGDIR="$mpldir" \
    PYTHON_BIN="$PYTHON_BIN" \
    ./run_blackbox_bottleneck_only_l7.sh
}

echo "=== Level-7 black-box ${BOTTLENECK_MODE} CR grid queue: ${QUEUE} ==="
echo "python=${PYTHON_BIN}"

case "$QUEUE" in
  gpu1)
    run_sweep 1 vsl 8 "0" "blackbox_vsl_l7_${BOTTLENECK_MODE}_cr8_logs" "/tmp/mpl-sdar-vsl-${BOTTLENECK_MODE}-cr8-gpu1"
    run_sweep 1 vsl 16 "0" "blackbox_vsl_l7_${BOTTLENECK_MODE}_cr16_logs" "/tmp/mpl-sdar-vsl-${BOTTLENECK_MODE}-cr16-gpu1"
    ;;
  gpu2)
    run_sweep 2 usl 8 "0" "blackbox_usl_l7_${BOTTLENECK_MODE}_cr8_logs" "/tmp/mpl-sdar-usl-${BOTTLENECK_MODE}-cr8-gpu2"
    run_sweep 2 usl 16 "0" "blackbox_usl_l7_${BOTTLENECK_MODE}_cr16_logs" "/tmp/mpl-sdar-usl-${BOTTLENECK_MODE}-cr16-gpu2"
    ;;
  gpu3)
    run_sweep 3 vsl 32 "0" "blackbox_vsl_l7_${BOTTLENECK_MODE}_cr32_logs" "/tmp/mpl-sdar-vsl-${BOTTLENECK_MODE}-cr32-gpu3"
    run_sweep 3 usl 32 "0" "blackbox_usl_l7_${BOTTLENECK_MODE}_cr32_logs" "/tmp/mpl-sdar-usl-${BOTTLENECK_MODE}-cr32-gpu3"
    ;;
  *)
    echo "Unknown queue: ${QUEUE}" >&2
    exit 2
    ;;
esac

echo
echo "[$(date '+%F %T')] Queue ${QUEUE} complete."
