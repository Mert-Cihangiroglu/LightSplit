# Shared settings for the UnSplit reproduction scripts. Source, do not run.
#
#   PYTHON     python with torch, torchvision, lpips                 default: python
#   OUT        output root                                           default: <attack-unsplit>/results/reproduce
#   DATA_ROOT  torchvision download directory for MNIST/F-MNIST     default: $OUT/<table>/data (downloaded there)
#   DEVICE     torch device passed to the demo                       default: cuda
#   CUDA_VISIBLE_DEVICES                                             default: 0
#   JOBS       cells run concurrently (each is small)                default: 2
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-$CODE_DIR/results/reproduce}"
DEVICE="${DEVICE:-cuda}"
JOBS="${JOBS:-2}"
# run <tag> <model_inversion_demo.py args...>
# Runs the demo inside $RUN_DIR (it writes results/ and data/ relative to its cwd), logs to
# $RUN_DIR/logs/<tag>.log, skips tags whose log already ends with the demo's "Results saved" line,
# and keeps at most $JOBS cells in flight.
run() { local tag="$1"; shift; local log="$RUN_DIR/logs/$tag.log"
  grep -qs "Results saved" "$log" && { echo "[skip] $tag"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || true; done
  echo "[$(date +%H:%M:%S)] $tag"
  ( cd "$RUN_DIR" && "$PY" "$CODE_DIR/model_inversion_demo.py" "$@" --device "$DEVICE" > "$log" 2>&1 \
      || echo "[fail] $tag (see $log)" ) & }
