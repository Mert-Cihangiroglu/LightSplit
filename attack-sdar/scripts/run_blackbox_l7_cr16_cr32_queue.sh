#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

QUEUE="${1:?Usage: $0 gpu1|gpu2|gpu3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_sweep() {
  local gpu="$1"
  local setting="$2"
  local cr="$3"
  local wcc_lambdas="$4"
  local logdir="$5"
  local mpldir="$6"

  echo
  echo "[$(date '+%F %T')] GPU ${gpu}: ${setting} CR=${cr} WCC={${wcc_lambdas}}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    SETTING="$setting" \
    MODEL=resnet \
    DATASET=cifar10 \
    LEVEL=7 \
    RUNS=1 \
    COMPRESSION_RATIO="$cr" \
    WCC_LAMBDAS="$wcc_lambdas" \
    LOGDIR="$logdir" \
    MPLCONFIGDIR="$mpldir" \
    PYTHON_BIN="$PYTHON_BIN" \
    ./run_blackbox_fixedrandom_only_l7.sh
}

echo "=== Level-7 black-box fixed-random CR=16/32 queue: ${QUEUE} ==="
echo "python=${PYTHON_BIN}"

case "$QUEUE" in
  gpu1)
    run_sweep 1 vsl 16 "0 0.01 0.1" blackbox_vsl_l7_cr16_logs /tmp/mpl-sdar-vsl-cr16-gpu1
    run_sweep 1 vsl 32 "0" blackbox_vsl_l7_cr32_logs /tmp/mpl-sdar-vsl-cr32-gpu1
    ;;
  gpu2)
    run_sweep 2 usl 16 "0 0.01 0.1" blackbox_usl_l7_cr16_logs /tmp/mpl-sdar-usl-cr16-gpu2
    run_sweep 2 usl 32 "0" blackbox_usl_l7_cr32_logs /tmp/mpl-sdar-usl-cr32-gpu2
    ;;
  gpu3)
    run_sweep 3 vsl 32 "0.01 0.1" blackbox_vsl_l7_cr32_logs /tmp/mpl-sdar-vsl-cr32-gpu3
    run_sweep 3 usl 32 "0.01 0.1" blackbox_usl_l7_cr32_logs /tmp/mpl-sdar-usl-cr32-gpu3
    ;;
  *)
    echo "Unknown queue: ${QUEUE}" >&2
    exit 2
    ;;
esac

echo
echo "[$(date '+%F %T')] Queue ${QUEUE} complete."
