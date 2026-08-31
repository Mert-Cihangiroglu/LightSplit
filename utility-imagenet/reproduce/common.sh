# Shared settings for the utility-imagenet reproduction scripts. Source, do not run.
#
#   IMAGENET_ROOT   ImageNet-1k root with train/ and val/ ImageFolder subdirectories (required unless FAKE=1)
#   PYTHON          python with torch + torchvision                     default: python
#   CUDA_VISIBLE_DEVICES  GPU to use                                    default: 0
#   OUT             output root                                         default: <utility-imagenet>/results/reproduce
#   EPOCHS / BATCH / WORKERS   training schedule; defaults reproduce the paper (90 epochs, batch 256)
#   ARMS            subset of arms to run (space separated), default all; runs are independent, so the
#                   arms can be split across GPUs by invoking the script once per GPU with a different ARMS
#   FAKE=1          synthetic data, for a pipeline smoke test only
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-$CODE_DIR/results/reproduce}"
EPOCHS="${EPOCHS:-90}"; BATCH="${BATCH:-256}"; WORKERS="${WORKERS:-8}"
if [ "${FAKE:-0}" = 1 ]; then DATA=(--fake-data); else DATA=(--data-root "${IMAGENET_ROOT:?set IMAGENET_ROOT or FAKE=1}"); fi
COMMON=("${DATA[@]}" --deep-head --num-clients 10 --batch-size "$BATCH" --num-workers "$WORKERS" --resume --gpu 0)
want() { [ -z "${ARMS:-}" ] || [[ " $ARMS " == *" $1 "* ]]; }
# run <arm-name> <train.py args...>  -> $OUT/<figure>/<arm>_{summary,epochs}.csv ; --resume skips finished arms
run() { local arm="$1"; shift; want "$arm" || return 0
  mkdir -p "$OUT/$FIG"; echo "[$(date +%H:%M:%S)] $FIG/$arm"
  "$PY" "$CODE_DIR/train.py" "${COMMON[@]}" "$@" --ckpt-dir "$OUT/$FIG/checkpoints/$arm" \
      --summary-csv "$OUT/$FIG/${arm}_summary.csv" --epoch-csv "$OUT/$FIG/${arm}_epochs.csv"; }
