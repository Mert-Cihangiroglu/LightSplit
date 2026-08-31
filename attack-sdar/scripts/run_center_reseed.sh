#!/usr/bin/env bash
# SDAR privacy LANE runner for CENTER LOSS (Wen et al. 2016), mirroring run_wcc_reseed.sh.
# Each job "cr:center:run" co-trains the SDAR victim with center loss on the fixed_random
# payload and reports reconstruction SSIM. Accuracy is NOT taken from here (the paper's
# accuracy is the 10-client deployed model); these runs are only for reconstruction SSIM.
# Resumable: skips a job whose log already has the SSIM line. tmux-safe.
set -uo pipefail
cd "$(dirname "$0")"
export TF_FORCE_GPU_ALLOW_GROWTH=true     # coexist on a shared card, no whole-GPU grab
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
GPU="${GPU:-2}"
LEVEL="${LEVEL:-7}"                        # 7 = deep 4096-d cut == deployed model
JOBS="${JOBS:-}"                          # space-separated "cr:center:run"
mkdir -p reseed_logs
echo "[$(date +%H:%M:%S)] lane GPU$GPU level=l$LEVEL jobs=[$JOBS]"
for job in $JOBS; do
  IFS=: read -r cr cl run <<< "$job"
  log="reseed_logs/usl_l${LEVEL}_cr${cr}_center${cl}_run${run}.log"
  if grep -q "Average SSIM" "$log" 2>/dev/null; then echo "[skip] $job (done)"; continue; fi
  echo "[$(date +%H:%M:%S)] GPU$GPU START l$LEVEL cr$cr center$cl run$run"
  CUDA_VISIBLE_DEVICES="$GPU" $PY run_sdar.py usl resnet cifar10 "$LEVEL" "$run" \
    --bottleneck_mode fixed_random --compression_ratio "$cr" --center_loss_lambda "$cl" \
    --print_to_stdout > "$log" 2>&1 || echo "[$(date +%H:%M:%S)] !! FAILED $job"
  ssim=$(grep "Average SSIM" "$log" | tail -1)
  acc=$(grep "Average label accuracy" "$log" | tail -1)
  echo "[$(date +%H:%M:%S)] GPU$GPU DONE cr$cr center$cl run$run | ${ssim:-no-ssim} | ${acc:-no-acc}"
done
echo "[$(date +%H:%M:%S)] GPU$GPU lane complete"
