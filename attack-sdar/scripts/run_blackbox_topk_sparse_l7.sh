#!/usr/bin/env bash
# ============================================================================
# SDAR black-box privacy sweep for the magnitude top-k SPARSIFICATION baseline
# (X5), at the cut-matched level-7 operating point.
#   method  : topk_sparse (keep the k_s largest-|h| coords/sample, zero the rest)
#   setting : vsl/cifar10, level 7 resnet cut (8x8x64 = 4096-d)
#   sweep   : CR in {8,16,32}  (matches the X5 utility table)
#   matched-byte: k_s = floor(4k/6) kept coords cost ~= k float32 values, so the
#     transmitted byte budget equals fixed_random/LightSplit at the same CR.
#
# NOTE: this is the TRUE magnitude sparsifier from the utility table.
#
# Long runs (~hours each). Launch DETACHED so it survives disconnect, e.g.
#   tmux new -s tks_l7 -d "GPU=0 ./run_blackbox_topk_sparse_l7.sh"
# Resume-safe: a finished run writes its history .npy; existing .npy -> skipped.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SETTING="${SETTING:-vsl}"
MODEL="${MODEL:-resnet}"
DATASET="${DATASET:-cifar10}"
LEVEL="${LEVEL:-7}"
RUNS="${RUNS:-0}"                       # run id(s), space-separated
CRS="${CRS:-8 16 32}"                   # matched-byte compression ratios
NUM_ITER_OVERRIDE="${NUM_ITER_OVERRIDE:-}"   # set small (e.g. 200) for a smoke run
PYTHON_BIN="${PYTHON_BIN:-python}"
LOGDIR="${LOGDIR:-blackbox_${SETTING}_l${LEVEL}_topk_sparse_logs}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
GPU="${GPU:-}"                          # optional: pin to one GPU id
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}   # conda CUDA (else TF sees 0 GPUs)
export TF_FORCE_GPU_ALLOW_GROWTH=true

mkdir -p "$LOGDIR"

echo "=== SDAR black-box topk_sparse level-${LEVEL} privacy sweep ==="
echo "setting=${SETTING} model=${MODEL} dataset=${DATASET} runs=${RUNS} CRs={${CRS}}"
echo "python=${PYTHON_BIN} gpu=${GPU:-default}"
[[ -n "$NUM_ITER_OVERRIDE" ]] && echo "num_iter_override=${NUM_ITER_OVERRIDE} (SMOKE)"
echo "logs=${LOGDIR}/"
echo

# output .npy path matching run_sdar.py's naming (param_tag = cr<CR>)
out_npy() {  # run_id cr
  echo "defense_results/topk_sparse/${SETTING}/${DATASET}/output/cr${2}_${MODEL}_l${LEVEL}_run${1}.npy"
}

run_one() {  # run_id cr
  local run_id="$1" cr="$2"
  local tag="topk_sparse_cr${cr}"
  local logfile="${LOGDIR}/${SETTING}_${MODEL}_${DATASET}_l${LEVEL}_run${run_id}_${tag}.log"
  local done_npy; done_npy="$(out_npy "$run_id" "$cr")"

  if [[ -f "$done_npy" ]]; then
    echo "[skip] ${tag} run=${run_id}: finished (${done_npy} exists)"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" run_sdar.py
    "$SETTING" "$MODEL" "$DATASET" "$LEVEL" "$run_id"
    --print_to_stdout
    --bottleneck_mode topk_sparse
    --compression_ratio "$cr"
  )
  [[ -n "$NUM_ITER_OVERRIDE" ]] && cmd+=(--num_iter_override "$NUM_ITER_OVERRIDE")

  echo "[$(date '+%F %T')] ${tag} run=${run_id} -> ${logfile}"
  printf '  command:'; printf ' %q' "${cmd[@]}"; printf '\n'
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL" "${cmd[@]}" 2>&1 | tee "$logfile"
  echo
}

for run_id in $RUNS; do
  for cr in $CRS; do
    run_one "$run_id" "$cr"
  done
done

echo "Done. Logs in ${LOGDIR}/  History/recon under defense_results/topk_sparse/${SETTING}/${DATASET}/"
