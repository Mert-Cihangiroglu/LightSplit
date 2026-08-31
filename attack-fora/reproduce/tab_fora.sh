#!/usr/bin/env bash
# Tables tab:fora-full (fora_full_appendix.tex), the FORA columns of tab:block_privacy_unsplit
# (block_privacy_attacks.tex) and of tab:privacy_all / tab:main-compact -- FORA against the CIFAR-10
# MobileNet cut at level 2 (d = 16384), 40 000 attack iterations per run.
# Cells: Raw, Raw+DP, Learned 1x1 (8/4/2 channels), LightSplit-F k in {2048,1024,512} x lambda_WCC in
# {0, 0.001 (k 2048), 0.01, 0.1}, top-k sparse, LightSplit-F+DP (sigma 0.02, clip = median payload norm),
# block-local projection b in {1,2,4} x k. Each cell runs the attacker seeds the paper averaged
# (seed offsets listed per cell in aggregate_fora.py: three for most cells, one for the lambda_WCC = 0.01
# rows, the 16x/32x lambda_WCC = 0.1 rows and the block rows). 61 runs of about 8 GPU-hours each; spread
# them with GPUS="0 1 2 3". Every run saves its 32-image reconstruction grids under recon_pics/ and its
# log under logs/; the aggregator scores the last grid of each run.
# Output: $OUT/tab_fora.csv
set -uo pipefail
source "$(dirname "$0")/common.sh"; TAB=tab_fora; RUN_DIR="$OUT/$TAB"; mkdir -p "$RUN_DIR/logs"
[ -n "${DATA_ROOT:-}" ] && ln -sfn "$DATA_ROOT" "$RUN_DIR/data"
S3="0 1 2"
SPECS=(   # "cell|seeds|main.py args"
  "raw|$S3|"
  "rawdp|$S3|--clip-norm 4.66 --noise-multiplier 0.02"
  "learned1x1_k8|0 1 2 3|--defense learned_1x1 --defense-k 8"
  "learned1x1_k4|$S3|--defense learned_1x1 --defense-k 4"
  "learned1x1_k2|$S3|--defense learned_1x1 --defense-k 2"
)
for k in 2048 1024 512; do SPECS+=("fixedrandom_k${k}_wcc0|$S3|--defense fixed_random --defense-k $k --wcc-lambda 0"); done
SPECS+=("fixedrandom_k2048_wcc0.001|$S3|--defense fixed_random --defense-k 2048 --wcc-lambda 0.001")
for k in 2048 1024 512; do SPECS+=("fixedrandom_k${k}_wcc0.01|0|--defense fixed_random --defense-k $k --wcc-lambda 0.01"); done
SPECS+=("fixedrandom_k2048_wcc0.1|$S3|--defense fixed_random --defense-k 2048 --wcc-lambda 0.1")
for k in 1024 512; do SPECS+=("fixedrandom_k${k}_wcc0.1|0|--defense fixed_random --defense-k $k --wcc-lambda 0.1"); done
SPECS+=("topksparse_k2048|$S3|--defense topk_sparse --defense-k 2048")
for k in 1024 512; do SPECS+=("topksparse_k${k}|0 1|--defense topk_sparse --defense-k $k"); done
declare -A CLIP=( [2048]=1.348 [1024]=0.942 [512]=0.672 )   # median payload norm of the untrained projection (measure_dp_norms.py)
for k in 2048 1024 512; do SPECS+=("dp_fixedrandom_k${k}|$S3|--defense fixed_random --defense-k $k --clip-norm ${CLIP[$k]} --noise-multiplier 0.02"); done
for b in 1 2 4; do for k in 2048 1024 512; do SPECS+=("blockrandom_b${b}_k${k}|0|--defense block_random --block-size $b --defense-k $k"); done; done
if [ "${SMOKE:-0}" = 1 ]; then   # pipeline check: three cells, 100 iterations, two seeds for the first
  ITERS=100; SPECS=("raw|0 1|" "fixedrandom_k2048_wcc0|0|--defense fixed_random --defense-k 2048 --wcc-lambda 0" "blockrandom_b2_k1024|0|--defense block_random --block-size 2 --defense-k 1024")
fi
run_specs
"$PY" "$REPRO_DIR/aggregate_fora.py" --root "$RUN_DIR" --out "$OUT/$TAB.csv"
