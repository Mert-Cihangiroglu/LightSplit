# Shared settings for the attack-sdar-cutwidth reproduction scripts. Source, do not run.
#
#   SDAR_TF_PYTHON  python with TensorFlow 2.9                          default: python
#   TF_LIB_DIR      if TensorFlow does not see the GPU, the env's lib dir (added to LD_LIBRARY_PATH)
#   CUDA_VISIBLE_DEVICES  GPU to use                                    default: 0
#   OUT             output root                                         default: <attack-sdar-cutwidth>/results/reproduce
#   SEEDS           default "0 1 2 3 4"
#   SDAR_ITERS      override the 20 000 attack iterations (smoke runs only)
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${SDAR_TF_PYTHON:-python}"
[ -n "${TF_LIB_DIR:-}" ] && export LD_LIBRARY_PATH="$TF_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3
OUT="${OUT:-$CODE_DIR/results/reproduce}"
SEEDS="${SEEDS:-0 1 2 3 4}"
# sdar_run <log-tag> <run_sdar.py args...> : one SDAR attack run; the log is what the aggregator reads.
# run_sdar.py must be started from scripts/ (it imports ../src). Skips a tag whose log already has an SSIM line.
sdar_run() { local tag="$1"; shift; local log="$OUT/$FIG/logs/$tag.log"; mkdir -p "$(dirname "$log")"
  grep -q "Average SSIM" "$log" 2>/dev/null && { echo "[skip] $tag"; return; }
  echo "[$(date +%H:%M:%S)] $FIG/$tag"
  local extra=(); [ -n "${SDAR_ITERS:-}" ] && extra=(--num_iter_override "$SDAR_ITERS")
  (cd "$CODE_DIR/scripts" && "$PY" run_sdar.py "$@" "${extra[@]}" --print_to_stdout > "$log" 2>&1) || echo "[failed] $tag (see $log)"; }
