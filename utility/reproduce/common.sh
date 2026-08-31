# Shared settings for the utility reproduction scripts. Source, do not run.
#
#   PYTHON      python with torch + torchvision                        default: python
#   DATA_ROOT   dataset cache (torchvision downloads CIFAR/FashionMNIST; GTSRB needs the folder layout data.py describes)
#                                                                        default: <utility>/data
#   CUDA_VISIBLE_DEVICES  GPU to use                                    default: 0
#   OUT         output root                                             default: <utility>/results/reproduce
#   SEEDS / EPOCHS   per script; defaults reproduce the paper
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-$CODE_DIR/results/reproduce}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"
# run <fig> <tag> <train_split.py args...>  -> $OUT/<fig>/<tag>_{summary,epochs}.csv ; skips finished tags
run() { local fig="$1" tag="$2"; shift 2; mkdir -p "$OUT/$fig"
  [ -s "$OUT/$fig/${tag}_summary.csv" ] && [ "$(wc -l < "$OUT/$fig/${tag}_summary.csv")" -ge 2 ] && { echo "[skip] $fig/$tag"; return; }
  echo "[$(date +%H:%M:%S)] $fig/$tag"
  "$PY" "$CODE_DIR/train_split.py" --data-root "$DATA_ROOT" "$@" --gpu 0 --no-viz \
      --summary-csv "$OUT/$fig/${tag}_summary.csv" --epoch-csv "$OUT/$fig/${tag}_epochs.csv"; }
