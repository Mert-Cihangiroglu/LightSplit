#!/usr/bin/env bash
# Table tab:block_privacy_grid (block-local projection under SDAR, CIFAR-10 USL) and the
# SDAR columns of tab:block_privacy_unsplit (the merged block table, incl. the +DP rows).
# Grid per level {7 = 4096-d, 4 = 8192-d}:
#   block_random b {1,2,4} x CR {8,16,32} x lambda_WCC {0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, 2 runs
#   b=8 (the global projection = LightSplit-F) at the same CR x lambda grid, 2 runs
#   +DP rows: b {1,2,4} at CR8, sigma 0.02, 2 runs (clips = median payload norms:
#     level 7 b1/b2/b4 = 20.120/20.502/20.476, b8 = 20.89;
#     level 4 b1/b2/b4 = 28.620/30.719/29.381, b8 = 30.87)
# 480 attacker runs, hours each on one GPU; LEVELS/CRS/BLOCKS/WCCS restrict the grid.
# Cells are reported as SSIM in % of the matched Raw, mean over runs, ungated
# (tab:block_privacy_grid caption); the Raw runs come from tab_sdar_all_split_points.sh.
set -uo pipefail
source "$(dirname "$0")/common.sh"
LEVELS="${LEVELS:-7 4}"
CRS="${CRS:-8 16 32}"
BLOCKS="${BLOCKS:-1 2 4 8}"
WCCS="${WCCS:-0 0.001 0.003 0.01 0.03 0.1}"
RUNS="${RUNS:-0 1}"
DP="${DP:-1}"
declare -A CLIPB1=( [7]=20.120 [4]=28.620 ) CLIPB2=( [7]=20.502 [4]=30.719 ) CLIPB4=( [7]=20.476 [4]=29.381 ) CLIPB8=( [7]=20.89 [4]=30.87 )

SPECS=""
for L in $LEVELS; do
  for cr in $CRS; do for b in $BLOCKS; do for w in $WCCS; do for r in $RUNS; do
    if [ "$b" = 8 ]; then
      SPECS+="usl_l${L}_blockgrid_b8_cr${cr}_wcc${w}_run${r}|usl resnet cifar10 $L $r|--bottleneck_mode fixed_random --compression_ratio $cr --wcc_lambda $w
"
    else
      SPECS+="usl_l${L}_blockgrid_b${b}_cr${cr}_wcc${w}_run${r}|usl resnet cifar10 $L $r|--bottleneck_mode block_random --block_size $b --compression_ratio $cr --wcc_lambda $w
"
    fi
  done; done; done; done
  if [ "$DP" = 1 ]; then
    for b in 1 2 4; do
      case "$b" in 1) clip=${CLIPB1[$L]};; 2) clip=${CLIPB2[$L]};; 4) clip=${CLIPB4[$L]};; esac
      for r in $RUNS; do
        SPECS+="usl_l${L}_blockdp_b${b}_cr8_run${r}|usl resnet cifar10 $L $r|--bottleneck_mode block_random --block_size $b --compression_ratio 8 --dp_clip_norm $clip --dp_sigma 0.02
"
      done
    done
  fi
done
printf '%s' "$SPECS" | run_specs
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_block_grid.py" --logs "$OUT" --out-dir "${CSV_OUT:-$REPRO_DIR/results}"
