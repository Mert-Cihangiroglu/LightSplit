#!/usr/bin/env bash
# ============================================================================
# SDAR black-box privacy sweep over the block_random locality dial (level 7).
#   method  : block_random (block-local fixed orthonormal projection)
#   setting : CR=8, level 7 resnet cut (8x8x64), vsl/cifar10
#   sweep   : block in {1,2,4,8}
#       block=1 -> per-position channel mix   (max utility, MIN privacy)
#       block=8 -> single global dense R == fixed_random (MAX privacy)
#   payload is held constant at 512 floats across all block sizes, so block is
#   the only moving variable -> the reconstruction-quality curve IS the result.
#
# Long runs (~hours each). Launch DETACHED so it survives disconnect, e.g.
#   tmux new -s br_l7 -d "GPU=0 ./run_blackbox_block_random_l7.sh"
# Resume-safe: a finished run writes its history .npy; existing .npy -> skipped.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SETTING="${SETTING:-vsl}"
MODEL="${MODEL:-resnet}"
DATASET="${DATASET:-cifar10}"
LEVEL="${LEVEL:-7}"
RUNS="${RUNS:-0}"                       # run id(s), space-separated
BLOCKS="${BLOCKS:-1 2 4 8}"             # the locality dial
COMPRESSION_RATIO="${COMPRESSION_RATIO:-8}"
NUM_ITER_OVERRIDE="${NUM_ITER_OVERRIDE:-}"   # set small (e.g. 200) for a smoke run
PYTHON_BIN="${PYTHON_BIN:-python}"
LOGDIR="${LOGDIR:-blackbox_${SETTING}_l${LEVEL}_block_random_logs}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
GPU="${GPU:-}"                          # optional: pin to one GPU id

mkdir -p "$LOGDIR"

echo "=== SDAR black-box block_random level-${LEVEL} privacy sweep ==="
echo "setting=${SETTING} model=${MODEL} dataset=${DATASET} runs=${RUNS} CR=${COMPRESSION_RATIO} blocks={${BLOCKS}}"
echo "python=${PYTHON_BIN} gpu=${GPU:-default}"
[[ -n "$NUM_ITER_OVERRIDE" ]] && echo "num_iter_override=${NUM_ITER_OVERRIDE} (SMOKE)"
echo "logs=${LOGDIR}/"
echo

# output .npy path matching run_sdar.py's naming (param_tag = cr<CR>_b<block>)
out_npy() {  # run_id block
  echo "defense_results/block_random/${SETTING}/${DATASET}/output/cr${COMPRESSION_RATIO}_b${2}_${MODEL}_l${LEVEL}_run${1}.npy"
}

run_one() {  # run_id block
  local run_id="$1" block="$2"
  local tag="block_random_cr${COMPRESSION_RATIO}_b${block}"
  local logfile="${LOGDIR}/${SETTING}_${MODEL}_${DATASET}_l${LEVEL}_run${run_id}_${tag}.log"
  local done_npy; done_npy="$(out_npy "$run_id" "$block")"

  if [[ -f "$done_npy" ]]; then
    echo "[skip] ${tag} run=${run_id}: finished (${done_npy} exists)"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" run_sdar.py
    "$SETTING" "$MODEL" "$DATASET" "$LEVEL" "$run_id"
    --print_to_stdout
    --bottleneck_mode block_random
    --compression_ratio "$COMPRESSION_RATIO"
    --block_size "$block"
  )
  [[ -n "$NUM_ITER_OVERRIDE" ]] && cmd+=(--num_iter_override "$NUM_ITER_OVERRIDE")

  echo "[$(date '+%F %T')] ${tag} run=${run_id} -> ${logfile}"
  printf '  command:'; printf ' %q' "${cmd[@]}"; printf '\n'
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL" "${cmd[@]}" 2>&1 | tee "$logfile"
  echo
}

for run_id in $RUNS; do
  for block in $BLOCKS; do
    run_one "$run_id" "$block"
  done
done

echo "Done. Logs in ${LOGDIR}/  History/recon under defense_results/block_random/${SETTING}/${DATASET}/"
