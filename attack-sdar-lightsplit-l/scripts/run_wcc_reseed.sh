#!/usr/bin/env bash
# Re-run the 3 noisy SDAR privacy cells with extra seeds (run ids 2,3,4) to replace the
# single-run values: 8x+wcc0.1 (non-physical outlier) and 32x+wcc{0.01,0.1} (diverged).
# SDAR sets no seed, so each run id is an independent trajectory. Accuracy is NOT taken
# from here (we use the 10-client deployed model); these runs are only for reconstruction SSIM.
# Resumable: skips a job whose log already has the SSIM line. tmux-safe.
set -uo pipefail
cd "$(dirname "$0")"
export TF_FORCE_GPU_ALLOW_GROWTH=true     # coexist on a shared card, no whole-GPU grab
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
GPU="${GPU:-2}"
LEVEL="${LEVEL:-4}"                        # SDAR split level (4=shallow 8192-d cut, 7=deep 4096-d cut == deployed)
JOBS="${JOBS:-}"                          # space-separated "cr:wcc:run"
mkdir -p reseed_logs
echo "[$(date +%H:%M:%S)] lane GPU$GPU level=l$LEVEL jobs=[$JOBS]"
for job in $JOBS; do
  IFS=: read -r cr wcc run <<< "$job"
  log="reseed_logs/usl_l${LEVEL}_cr${cr}_wcc${wcc}_run${run}.log"
  if grep -q "Average SSIM" "$log" 2>/dev/null; then echo "[skip] $job (done)"; continue; fi
  echo "[$(date +%H:%M:%S)] GPU$GPU START l$LEVEL cr$cr wcc$wcc run$run (20000 iter)"
  CUDA_VISIBLE_DEVICES="$GPU" $PY run_sdar.py usl resnet cifar10 "$LEVEL" "$run" \
    --bottleneck_mode fixed_random --compression_ratio "$cr" --wcc_lambda "$wcc" \
    --print_to_stdout > "$log" 2>&1 || echo "[$(date +%H:%M:%S)] !! FAILED $job"
  ssim=$(grep "Average SSIM" "$log" | tail -1)
  acc=$(grep "Average label accuracy" "$log" | tail -1)
  echo "[$(date +%H:%M:%S)] GPU$GPU DONE cr$cr wcc$wcc run$run | ${ssim:-no-ssim} | ${acc:-no-acc}"
done
echo "[$(date +%H:%M:%S)] GPU$GPU lane complete"
