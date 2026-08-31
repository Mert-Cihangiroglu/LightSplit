#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LEVEL="${1:?Usage: $0 level run vsl|usl|both}"
RUN_ID="${2:?Usage: $0 level run vsl|usl|both}"
WHICH="${3:?Usage: $0 level run vsl|usl|both}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-resnet}"
DATASET="${DATASET:-cifar10}"
LOGDIR="${LOGDIR:-blackbox_l${LEVEL}_raw_rerun_logs}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

mkdir -p "$LOGDIR"

run_raw() {
  local setting="$1"
  local logfile="${LOGDIR}/${setting}_${MODEL}_${DATASET}_l${LEVEL}_run${RUN_ID}_raw.log"
  local cmd=(
    "$PYTHON_BIN" run_sdar.py
    "$setting" "$MODEL" "$DATASET" "$LEVEL" "$RUN_ID"
    --print_to_stdout
  )

  echo
  echo "[$(date '+%F %T')] GPU ${GPU}: raw ${setting} level=${LEVEL} run=${RUN_ID} -> ${logfile}"
  printf '  command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL" "${cmd[@]}" 2>&1 | tee "$logfile"
}

echo "=== SDAR raw level-${LEVEL} rerun: run=${RUN_ID}, which=${WHICH} ==="
echo "python=${PYTHON_BIN}"
echo "logs=${LOGDIR}/"

case "$WHICH" in
  vsl)
    run_raw vsl
    ;;
  usl)
    run_raw usl
    ;;
  both)
    run_raw vsl
    run_raw usl
    ;;
  *)
    echo "Unknown raw target: ${WHICH}; expected vsl, usl, or both." >&2
    exit 2
    ;;
esac

echo
echo "[$(date '+%F %T')] Raw rerun ${WHICH} complete."
