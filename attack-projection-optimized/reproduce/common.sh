# Shared settings for the projection-optimized attack reproduction scripts. Source, do not run.
#
#   PYTHON      python with torch + torchvision (and lpips for the aggregator)   default: python
#   DATA_ROOT   dataset cache (torchvision downloads CIFAR-10)                   default: <folder>/data
#   CUDA_VISIBLE_DEVICES  GPU to use                                             default: 0
#   OUT         output root                                                      default: <folder>/reproduce/results
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-$REPRO_DIR/results}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"
