#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
VSL_GPU="${VSL_GPU:-2}"
USL_GPU="${USL_GPU:-1}"
LEVEL="${LEVEL:-4}"
RUNS="${RUNS:-1}"
CRS="${CRS:-32}"
MODES="${MODES:-learned_1x1}"
ROOT_LOGDIR="${ROOT_LOGDIR:-blackbox_l4_learned_topk_cr32_logs}"

mkdir -p "$ROOT_LOGDIR"

run_setting_queue() {
  local setting="$1"
  local gpu="$2"

  echo "[$(date '+%F %T')] Start setting=${setting} level=${LEVEL} gpu=${gpu}"
  echo "python=${PYTHON_BIN}"
  echo "modes=${MODES}"
  echo "crs=${CRS}"
  echo

  for mode in $MODES; do
    for cr in $CRS; do
      local logdir="blackbox_${setting}_l${LEVEL}_${mode}_cr${cr}_logs"
      local mpldir="/tmp/mpl-sdar-${setting}-l${LEVEL}-${mode}-cr${cr}-gpu${gpu}"

      echo "[$(date '+%F %T')] GPU ${gpu}: ${setting} ${mode} CR=${cr} WCC={0}"
      CUDA_VISIBLE_DEVICES="$gpu" \
        SETTING="$setting" \
        MODEL=resnet \
        DATASET=cifar10 \
        LEVEL="$LEVEL" \
        RUNS="$RUNS" \
        BOTTLENECK_MODE="$mode" \
        COMPRESSION_RATIO="$cr" \
        WCC_LAMBDAS="0" \
        LOGDIR="$logdir" \
        MPLCONFIGDIR="$mpldir" \
        PYTHON_BIN="$PYTHON_BIN" \
        ./run_blackbox_bottleneck_only.sh
      echo
    done
  done

  echo "[$(date '+%F %T')] Complete setting=${setting} level=${LEVEL} gpu=${gpu}"
}

echo "=== SDAR black-box level-${LEVEL} learned 1x1 + SVD-rank-k CR 32 queue ==="
echo "VSL gpu=${VSL_GPU}; USL gpu=${USL_GPU}"
echo "root logs=${ROOT_LOGDIR}/"
echo

run_setting_queue vsl "$VSL_GPU" > "${ROOT_LOGDIR}/vsl_queue.out" 2>&1 &
vsl_pid="$!"
run_setting_queue usl "$USL_GPU" > "${ROOT_LOGDIR}/usl_queue.out" 2>&1 &
usl_pid="$!"

echo "vsl_queue_pid=${vsl_pid}"
echo "usl_queue_pid=${usl_pid}"

wait "$vsl_pid"
wait "$usl_pid"

echo "[$(date '+%F %T')] All level-${LEVEL} learned_1x1 CR 32 queues complete."
