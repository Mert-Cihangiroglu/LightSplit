#!/usr/bin/env bash
# X4-privacy: SDAR reconstruction against DP-defended victims at sigma=0.02.
# Four cells x runs {1,2,3}, cut-matched at L7 (4096-d cut == deployed):
#   raw        : no defense (undefended ceiling)
#   rawdp      : raw + DP  (clip=median raw payload norm, sigma=0.02)
#   cr8        : fixed_random CR8 (LightSplit-F, no DP)  == X1 lambda=0 baseline
#   cr8dp      : fixed_random CR8 + DP  (clip=median CR8 payload norm, sigma=0.02)
# Clip = per-config MEDIAN ||payload||_2 measured by measure_dp_norms.py (untrained f, L7):
#   raw median ~62.75 ; fixed_random CR8 median ~20.89.
# SDAR co-trains its own victim; each run id is an independent trajectory (no seed set).
# Read privacy (SSIM) only from HEALTHY victims (label acc >= 0.40); a diverged victim's
# reconstruction is meaningless. Resumable: skips a job whose log already has "Average SSIM".
set -uo pipefail
cd "$(dirname "$0")"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
GPU="${GPU:-0}"
LEVEL="${LEVEL:-7}"
SIGMA="${SIGMA:-0.02}"
CLIP_RAW="${CLIP_RAW:-62.75}"
CLIP_CR8="${CLIP_CR8:-20.89}"
JOBS="${JOBS:-}"                 # space-separated "cell:run", cell in {raw,rawdp,cr8,cr8dp}
mkdir -p reseed_logs
echo "[$(date +%H:%M:%S)] DP lane GPU$GPU level=l$LEVEL sigma=$SIGMA jobs=[$JOBS]"
for job in $JOBS; do
  IFS=: read -r cell run <<< "$job"
  case "$cell" in
    raw)   tag="raw";              args="";;
    rawdp) tag="raw_dps${SIGMA}";  args="--dp_clip_norm $CLIP_RAW --dp_sigma $SIGMA";;
    cr8)   tag="cr8";              args="--bottleneck_mode fixed_random --compression_ratio 8";;
    cr8dp) tag="cr8_dps${SIGMA}";  args="--bottleneck_mode fixed_random --compression_ratio 8 --dp_clip_norm $CLIP_CR8 --dp_sigma $SIGMA";;
    *) echo "[skip] unknown cell '$cell'"; continue;;
  esac
  log="reseed_logs/usl_l${LEVEL}_${tag}_run${run}.log"
  if grep -q "Average SSIM" "$log" 2>/dev/null; then echo "[skip] $job (done)"; continue; fi
  echo "[$(date +%H:%M:%S)] GPU$GPU START $cell run$run -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" $PY run_sdar.py usl resnet cifar10 "$LEVEL" "$run" \
    $args --print_to_stdout > "$log" 2>&1 || echo "[$(date +%H:%M:%S)] !! FAILED $job"
  ssim=$(grep "Average SSIM" "$log" | tail -1)
  acc=$(grep "Average label accuracy" "$log" | tail -1)
  echo "[$(date +%H:%M:%S)] GPU$GPU DONE $cell run$run | ${ssim:-no-ssim} | ${acc:-no-acc}"
done
echo "[$(date +%H:%M:%S)] GPU$GPU DP lane complete"
