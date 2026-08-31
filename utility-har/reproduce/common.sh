# Shared settings for the utility-har reproduction scripts. Source, do not run.
#
#   PYTHON     python with torch + scikit-learn                     default: python
#   OUT        output root                                          default: <utility-har>/results/reproduce
#   SEEDS      default "0 1 2";  EPOCHS default 120
#   HAR_DATA   where UCI HAR is downloaded and cached (read by train_har.py and attacks.py); default <utility-har>/data
#   The models are small MLPs; the scripts run on CPU (CUDA_VISIBLE_DEVICES="" by default).
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
OUT="${OUT:-$CODE_DIR/results/reproduce}"
SEEDS="${SEEDS:-0 1 2}"; EPOCHS="${EPOCHS:-120}"
DATA=(); [ -n "${HAR_DATA:-}" ] && DATA=(--data_root "$HAR_DATA")
# cell <liftback> <wcc> <k|none> <seed>: train, then run the three attacks; skips finished cells
cell() { local lb="$1" w="$2" k="$3" s="$4" tag kargs=()
  if [ "$k" = none ]; then tag="${lb}_w${w}_none_s${s}"; else tag="${lb}_w${w}_k${k}_s${s}"; kargs=(--k "$k"); fi
  local dir="$OUT/$FIG/$tag"
  [ -f "$dir/attacks.json" ] && { echo "[skip] $tag"; return; }
  echo "[$(date +%H:%M:%S)] $FIG/$tag"
  "$PY" "$CODE_DIR/train_har.py" --liftback "$lb" --wcc_lambda "$w" "${kargs[@]}" --d 256 \
      --epochs "$EPOCHS" --seed "$s" "${DATA[@]}" --out "$dir"
  "$PY" "$CODE_DIR/attacks.py" --run "$dir" --seed "$s"; }
