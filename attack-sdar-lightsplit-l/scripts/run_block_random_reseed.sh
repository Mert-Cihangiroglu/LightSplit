#!/usr/bin/env bash
# Per-lane block_random SDAR privacy runner (the locality dial). Jobs are "cr:block:run".
# block_random has NO WCC, so it is projection-only (more stable than the WCC sweep).
# Accuracy is recorded only to gate healthy vs collapsed victims; the paper's accuracy
# comes from the utility harness. Resume-guarded by the SSIM line in the log. tmux-safe.
set -uo pipefail
cd "$(dirname "$0")"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}   # conda CUDA 11.2/cuDNN 8.1 (else TF sees 0 GPUs)
PY=python
GPU="${GPU:-2}"
LEVEL="${LEVEL:-7}"
SETTING="${SETTING:-vsl}"
JOBS="${JOBS:-}"                          # space-separated "cr:block:run"
mkdir -p reseed_logs
echo "[$(date +%H:%M:%S)] block_random lane GPU$GPU setting=$SETTING level=l$LEVEL jobs=[$JOBS]"
for job in $JOBS; do
  IFS=: read -r cr block run wcc <<< "$job"
  wcc="${wcc:-0}"                                   # optional 4th field; default = projection-only
  wtag=""; [ "$wcc" != "0" ] && wtag="_wcc${wcc}"   # match run_sdar.py: only tag the log when wcc!=0
  log="reseed_logs/${SETTING}_l${LEVEL}_blockrandom_cr${cr}_b${block}${wtag}_run${run}.log"
  if grep -q "Average SSIM" "$log" 2>/dev/null; then echo "[skip] $job (done)"; continue; fi
  echo "[$(date +%H:%M:%S)] GPU$GPU START $SETTING cr$cr b$block wcc$wcc run$run (20000 iter)"
  CUDA_VISIBLE_DEVICES="$GPU" $PY -u run_sdar.py "$SETTING" resnet cifar10 "$LEVEL" "$run" \
    --bottleneck_mode block_random --compression_ratio "$cr" --block_size "$block" \
    --wcc_lambda "$wcc" --print_to_stdout > "$log" 2>&1 || echo "[$(date +%H:%M:%S)] !! FAILED $job"
  ssim=$(grep "Average SSIM" "$log" | tail -1)
  acc=$(grep "Average label accuracy" "$log" | tail -1)
  echo "[$(date +%H:%M:%S)] GPU$GPU DONE $SETTING cr$cr b$block wcc$wcc run$run | ${ssim:-no-ssim} | ${acc:-no-acc}"
done
echo "[$(date +%H:%M:%S)] GPU$GPU lane complete"
