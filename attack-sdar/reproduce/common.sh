# Shared settings for the SDAR reproduction drivers. Source, do not run.
#
#   PYTHON   python of the TensorFlow environment (lightsplit-tf in the README)   default: python
#   GPUS     space-separated GPU ids; runs are distributed round-robin            default: "0"
#   OUT      log root                                                             default: <attack-sdar>/reproduce/logs
#
# run_sdar.py resolves src/ relative to scripts/, so every run executes from
# inside scripts/. Reconstruction grids and histories land in the upstream
# layout under scripts/ (sdar_results/, defense_results/); the aggregators read
# the run log plus the grid the log points to.
#
# TensorFlow only sees the GPU when the conda env's own CUDA libraries are on
# LD_LIBRARY_PATH; otherwise it silently trains on the CPU. We derive it from
# PYTHON when it points into a conda env - check nvidia-smi after launching.
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
OUT="${OUT:-$REPRO_DIR/logs}"
mkdir -p "$OUT"
case "$PY" in
  */envs/*/bin/python*) export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";;
esac
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3
GPUS="${GPUS:-0}"

# run_one <gpu> <tag> <run_sdar.py args...>   log: $OUT/<tag>.log ; skips finished tags
run_one() { local gpu="$1" tag="$2"; shift 2
  local log="$OUT/${tag}.log"
  if grep -q "Average SSIM" "$log" 2>/dev/null; then echo "[skip] $tag"; return 0; fi
  echo "[$(date +%H:%M:%S)] GPU$gpu $tag"
  ( cd "$CODE_DIR/scripts" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_sdar.py "$@" --print_to_stdout ) \
    > "$log" 2>&1 || echo "[fail] $tag (see $log)"
}

# run_specs: SPECS lines "tag|setting level run|extra args..." distributed over $GPUS,
# one job per GPU lane at a time.
run_specs() {
  local -a lanes=($GPUS) pids=()
  local i=0
  while IFS='|' read -r tag pos extra; do
    [ -z "$tag" ] && continue
    local gpu="${lanes[$((i % ${#lanes[@]}))]}"; i=$((i+1))
    (
      # serialize per lane: wait for this lane's previous job
      run_one "$gpu" "$tag" $pos $extra
    ) &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge "${#lanes[@]}" ]; do wait -n || true; done
  done
  wait || true
}
