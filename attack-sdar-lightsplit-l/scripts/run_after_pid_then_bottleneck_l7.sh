#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?Usage: $0 wait_pid kill_pids gpu setting cr logdir mpldir}"
KILL_PIDS="${2:-}"
GPU="${3:?Usage: $0 wait_pid kill_pids gpu setting cr logdir mpldir}"
SETTING="${4:?Usage: $0 wait_pid kill_pids gpu setting cr logdir mpldir}"
COMPRESSION_RATIO="${5:?Usage: $0 wait_pid kill_pids gpu setting cr logdir mpldir}"
LOGDIR="${6:?Usage: $0 wait_pid kill_pids gpu setting cr logdir mpldir}"
MPLDIR="${7:?Usage: $0 wait_pid kill_pids gpu setting cr logdir mpldir}"
BOTTLENECK_MODE="${BOTTLENECK_MODE:-learned_1x1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")"

echo "[$(date '+%F %T')] Waiting for PID ${WAIT_PID} before ${SETTING} ${BOTTLENECK_MODE} CR=${COMPRESSION_RATIO} WCC=0 on GPU ${GPU}."

while true; do
  stat="$(ps -o stat= -p "$WAIT_PID" 2>/dev/null | awk '{print $1}')"
  if [[ -z "$stat" || "$stat" == Z* ]]; then
    break
  fi
  sleep 30
done

if [[ -n "$KILL_PIDS" ]]; then
  echo "[$(date '+%F %T')] Stopping old paused wrappers: ${KILL_PIDS}"
  # shellcheck disable=SC2086
  kill $KILL_PIDS 2>/dev/null || true
  sleep 2
  # shellcheck disable=SC2086
  kill -KILL $KILL_PIDS 2>/dev/null || true
fi

echo "[$(date '+%F %T')] Starting ${SETTING} ${BOTTLENECK_MODE} CR=${COMPRESSION_RATIO} WCC=0 on GPU ${GPU}."
CUDA_VISIBLE_DEVICES="$GPU" \
  SETTING="$SETTING" \
  MODEL=resnet \
  DATASET=cifar10 \
  LEVEL=7 \
  RUNS=1 \
  BOTTLENECK_MODE="$BOTTLENECK_MODE" \
  COMPRESSION_RATIO="$COMPRESSION_RATIO" \
  WCC_LAMBDAS=0 \
  LOGDIR="$LOGDIR" \
  MPLCONFIGDIR="$MPLDIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  ./run_blackbox_bottleneck_only_l7.sh

echo "[$(date '+%F %T')] Done."
