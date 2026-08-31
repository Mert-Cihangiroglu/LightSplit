#!/usr/bin/env bash
# Fill fixed_random (= b8, the global-R endpoint) USL SSIM to full 6-lambda x 2-seed, matching
# block_random b1/b2/b4. fixed_random is global by construction (no block_size), so this is the
# TRUE global endpoint at each level (L4 = global 16x16, not block=8). Both levels L7 + L4.
# Logs go to blackbox_usl_l<L>_cr<cr>_logs/ under the standard per-run naming.
# Resume-guarded (skip if the log already has an SSIM line). Detach: tmux new -s frq -d "bash run_fixed_random_usl_fill.sh".
set -uo pipefail
cd "$(dirname "$0")"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
PY=python
LEVELS="${LEVELS:-7 4}"
CRS="${CRS:-8 16 32}"
WCCS="${WCCS:-0 0.001 0.003 0.01 0.03 0.1}"
RUNS="${RUNS:-0 1}"
GPUS=(${GPUS:-0 1 2 3})
PER_GPU="${PER_GPU:-2}"
log(){ echo "[frq $(date '+%F %T')] $*"; }

logpath(){ # L cr wcc seed
  local w=""; [ "$3" != "0" ] && w="_wcc$3"
  echo "blackbox_usl_l$1_cr$2_logs/usl_resnet_cifar10_l$1_run$4_fixed_random_cr$2${w}.log"
}

# build the pending job list (skip cells whose log already has an SSIM line)
JOBS=()
for L in $LEVELS; do for cr in $CRS; do for w in $WCCS; do for r in $RUNS; do
  lg=$(logpath "$L" "$cr" "$w" "$r")
  grep -q "Average SSIM" "$lg" 2>/dev/null && continue
  JOBS+=("$L:$cr:$w:$r")
done; done; done; done
log "fixed_random USL fill: ${#JOBS[@]} pending jobs (levels $LEVELS, seeds $RUNS)"
[ ${#JOBS[@]} -eq 0 ] && { log "nothing to do"; exit 0; }

# lanes = GPUS x PER_GPU; assign jobs round-robin
lanes=(); for g in "${GPUS[@]}"; do for ((i=0;i<PER_GPU;i++)); do lanes+=("$g"); done; done
nl=${#lanes[@]}
declare -A LJ
for idx in "${!JOBS[@]}"; do LJ[$((idx % nl))]+="${JOBS[$idx]} "; done

for l in "${!lanes[@]}"; do
  g=${lanes[$l]}; jl="${LJ[$l]:-}"; [ -z "$jl" ] && continue
  sess="frq_n${l}g${g}"
  log "$sess gpu$g <- $jl"
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new-session -d -s "$sess" -c "$PWD" "
    export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
    for j in $jl; do
      IFS=: read -r L cr w r <<< \"\$j\"
      wtag=''; [ \"\$w\" != '0' ] && wtag=\"_wcc\$w\"
      lg=blackbox_usl_l\${L}_cr\${cr}_logs/usl_resnet_cifar10_l\${L}_run\${r}_fixed_random_cr\${cr}\${wtag}.log
      mkdir -p blackbox_usl_l\${L}_cr\${cr}_logs
      grep -q 'Average SSIM' \"\$lg\" 2>/dev/null && { echo skip \$j; continue; }
      echo \"[\$(date '+%T')] START L\$L cr\$cr wcc\$w run\$r\"
      CUDA_VISIBLE_DEVICES=$g $PY -u run_sdar.py usl resnet cifar10 \$L \$r --print_to_stdout \
        --bottleneck_mode fixed_random --compression_ratio \$cr --wcc_lambda \$w 2>&1 | tee \"\$lg\"
    done
    echo '[frq] lane $sess done'
  "
done
log "launched ${nl} lanes."
