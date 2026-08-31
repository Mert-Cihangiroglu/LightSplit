#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SETTING="${SETTING:-vsl}"
MODEL="${MODEL:-resnet}"
DATASET="${DATASET:-cifar10}"
LEVEL="${LEVEL:-6}"
RUNS="${RUNS:-1}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-8}"
WCC_LAMBDAS="${WCC_LAMBDAS:-0 0.01 0.1}"
NUM_ITER_OVERRIDE="${NUM_ITER_OVERRIDE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOGDIR="${LOGDIR:-blackbox_${SETTING}_l${LEVEL}_cr${COMPRESSION_RATIO}_logs}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

mkdir -p "$LOGDIR"

echo "=== SDAR black-box fixed-random level-${LEVEL} sweep ==="
echo "setting=${SETTING} model=${MODEL} dataset=${DATASET} runs=${RUNS} compression_ratio=${COMPRESSION_RATIO} wcc_lambdas=${WCC_LAMBDAS}"
echo "python=${PYTHON_BIN}"
if [[ -n "$NUM_ITER_OVERRIDE" ]]; then
  echo "num_iter_override=${NUM_ITER_OVERRIDE}"
fi
echo "logs=${LOGDIR}/"
echo

run_one() {
  local run_id="$1"
  local wcc_lambda="$2"

  local tag="fixed_random_cr${COMPRESSION_RATIO}_wcc${wcc_lambda}"
  local logfile="${LOGDIR}/${SETTING}_${MODEL}_${DATASET}_l${LEVEL}_run${run_id}_${tag}.log"
  local cmd=(
    "$PYTHON_BIN" run_sdar.py
    "$SETTING" "$MODEL" "$DATASET" "$LEVEL" "$run_id"
    --print_to_stdout
    --bottleneck_mode fixed_random
    --compression_ratio "$COMPRESSION_RATIO"
    --wcc_lambda "$wcc_lambda"
  )

  if [[ -n "$NUM_ITER_OVERRIDE" ]]; then
    cmd+=(--num_iter_override "$NUM_ITER_OVERRIDE")
  fi

  echo "[$(date '+%F %T')] ${tag} run=${run_id} -> ${logfile}"
  printf '  command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  PYTHONUNBUFFERED="$PYTHONUNBUFFERED" TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL" "${cmd[@]}" 2>&1 | tee "$logfile"
  echo
}

for run_id in $RUNS; do
  for wcc_lambda in $WCC_LAMBDAS; do
    run_one "$run_id" "$wcc_lambda"
  done
done

echo "Done. Logs are in ${LOGDIR}/"
