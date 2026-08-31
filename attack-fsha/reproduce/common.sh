# Shared settings for the attack-fsha reproduction scripts. Source, do not run.
#
#   FSHA_TF_PYTHON     python with TensorFlow 2.9 (trains the attack)        default: python
#   FSHA_TORCH_PYTHON  python with torch + lpips (scores reconstructions)     default: python
#   TF_LIB_DIR         if TensorFlow does not see the GPU, the env's lib dir (added to LD_LIBRARY_PATH)
#   CUDA_VISIBLE_DEVICES  GPU to use                                          default: 0
#   OUT                output root                                            default: <attack-fsha>/results/reproduce
#   SEEDS / DATASETS / ITERATIONS / CLEAN_EPOCHS / LOG_FREQ   the grid; defaults reproduce the paper
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSHA_DIR="${FSHA_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
TF_PY="${FSHA_TF_PYTHON:-python}"
TORCH_PY="${FSHA_TORCH_PYTHON:-python}"
[ -n "${TF_LIB_DIR:-}" ] && export LD_LIBRARY_PATH="$TF_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3
OUT="${OUT:-$FSHA_DIR/results/reproduce}"
SEEDS="${SEEDS:-123 124 125}"
DATASETS="${DATASETS:-mnist fashion_mnist}"
ITERATIONS="${ITERATIONS:-10000}"
CLEAN_EPOCHS="${CLEAN_EPOCHS:-5}"
LOG_FREQ="${LOG_FREQ:-500}"
COMMON=(--iterations "$ITERATIONS" --log-frequency "$LOG_FREQ" --batch-size 64 --id-setup 3 --sample-count 20 --keep-going)
done_if() { [ -f "$1" ] && { echo "[skip] $(dirname "$1") done"; return 0; }; return 1; }
