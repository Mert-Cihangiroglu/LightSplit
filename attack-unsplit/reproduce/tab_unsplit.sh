#!/usr/bin/env bash
# Table tab:unsplit_attack_full (unsplit_full.tex), the UnSplit columns of tab:block_privacy_unsplit
# (block_privacy_attacks.tex) and of tab:privacy_all / tab:main-compact -- UnSplit model inversion
# against MnistNet at cuts 2 and 4, MNIST and Fashion-MNIST. Per (dataset, cut): Raw, Learned 1x1 x
# CR{8,16,32}, LightSplit-F x lambda{0,0.01,0.1} x CR{8,16,32}, top-k sparse x CR{8,16,32}, and the DP
# rows (sigma 0.02, 10 images, seed 0: Raw+DP at both cuts, LightSplit-F+DP at CR 8, at cut 4 as the
# global b=8 tile); at cut 2 also the block-local projection b{1,2,4} x CR and b{1,2,4}+DP at CR 8.
# Every victim trains 20 epochs (Adam 1e-3, AMSGrad, batch 64); the attack runs upstream's
# 1000 outer / 100 input / 100 model iterations on one image per cell (the paper's single-image cells
# are unseeded, so they vary run to run). 88 single-image cells + 10 ten-image cells; a single-image
# cell takes a few minutes on one GPU, a ten-image cell ~10x. RSEEDS="0 1 2 3" adds the extra R seeds
# of the (cut 4, CR 8, lambda 0.1) cell.
# Output: $OUT/tab_unsplit.csv (all cells) and $OUT/tab_unsplit_headline.csv (split-2 summary)
set -uo pipefail
source "$(dirname "$0")/common.sh"; TAB=tab_unsplit; RUN_DIR="$OUT/$TAB"; mkdir -p "$RUN_DIR/logs"
[ -n "${DATA_ROOT:-}" ] && ln -sfn "$DATA_ROOT" "$RUN_DIR/data"
DATASETS="${DATASETS:-mnist f_mnist}"; SPLITS="${SPLITS:-2 4}"; CRS="${CRS:-8 16 32}"; WCCS="${WCCS:-0 0.01 0.1}"
BLOCKS="${BLOCKS:-1 2 4}"; RSEEDS="${RSEEDS:-}"; EPOCHS="${EPOCHS:-20}"; NIMG_DP="${NIMG_DP:-10}"
ITERS=(--main-iters "${MAIN_ITERS:-1000}" --input-iters "${INPUT_ITERS:-100}" --model-iters "${MODEL_ITERS:-100}")
if [ "${SMOKE:-0}" = 1 ]; then   # pipeline check only: one dataset, cut 2, tiny budgets
  DATASETS=mnist; SPLITS="2 4"; CRS=8; WCCS=0; BLOCKS=2; RSEEDS=""; EPOCHS=1; NIMG_DP=2
  ITERS=(--main-iters 3 --input-iters 2 --model-iters 2 --max-train-batches 20)
fi
# DP clip per configuration = median payload L2 norm (see measure_dp_norms.py); noise std = sigma x clip
declare -A CLIP=( [mnist_s2_raw]=5.249 [mnist_s2_fixed_random]=1.845 [f_mnist_s2_raw]=6.751 [f_mnist_s2_fixed_random]=2.407
                  [mnist_s4_raw]=3.713 [mnist_s4_b8]=1.077 [f_mnist_s4_raw]=2.666 [f_mnist_s4_b8]=1.497
                  [mnist_b1]=3.111 [mnist_b2]=2.982 [mnist_b4]=2.315 [f_mnist_b1]=1.485 [f_mnist_b2]=3.094 [f_mnist_b4]=3.314 )
for ds in $DATASETS; do for sp in $SPLITS; do
  base=("$ds" "$sp" 1 --epochs "$EPOCHS" "${ITERS[@]}")
  run "${ds}_s${sp}_none" "${base[@]}" --defense none
  for cr in $CRS; do
    run "${ds}_s${sp}_learned_1x1_cr${cr}" "${base[@]}" --defense learned_1x1 --cr "$cr"
    run "${ds}_s${sp}_topk_sparse_cr${cr}" "${base[@]}" --defense topk_sparse --cr "$cr"
    for w in $WCCS; do run "${ds}_s${sp}_fixed_random_cr${cr}_wcc${w}" "${base[@]}" --defense fixed_random --cr "$cr" --wcc-lambda "$w"; done
    if [ "$sp" = 2 ]; then for b in $BLOCKS; do
      run "${ds}_s2_block${b}_cr${cr}" "${base[@]}" --defense block_random --block-size "$b" --cr "$cr" --wcc-lambda 0; done; fi
  done
  if [ "$sp" = 4 ]; then for s in $RSEEDS; do
    run "${ds}_s4_fixed_random_cr8_wcc0.1_seed${s}" "${base[@]}" --defense fixed_random --cr 8 --wcc-lambda 0.1 --seed "$s"; done; fi
  dp=("$ds" "$sp" "$NIMG_DP" --epochs "$EPOCHS" "${ITERS[@]}" --noise-multiplier 0.02 --seed 0)
  run "${ds}_s${sp}_raw_dp0.02_n${NIMG_DP}" "${dp[@]}" --defense raw --cr 8 --wcc-lambda 0 --clip-norm "${CLIP[${ds}_s${sp}_raw]}"
  if [ "$sp" = 2 ]; then
    run "${ds}_s2_raw_dp0_n${NIMG_DP}" "$ds" 2 "$NIMG_DP" --epochs "$EPOCHS" "${ITERS[@]}" --defense raw --cr 8 --wcc-lambda 0 --clip-norm 0 --noise-multiplier 0 --seed 0
    run "${ds}_s2_fixed_random_cr8_dp0.02_n${NIMG_DP}" "${dp[@]}" --defense fixed_random --cr 8 --wcc-lambda 0 --clip-norm "${CLIP[${ds}_s2_fixed_random]}"
    for b in $BLOCKS; do run "${ds}_s2_block${b}_cr8_dp0.02" "$ds" 2 1 --epochs "$EPOCHS" "${ITERS[@]}" --defense block_random --block-size "$b" --cr 8 --wcc-lambda 0 \
        --clip-norm "${CLIP[${ds}_b${b}]}" --noise-multiplier 0.02 --seed 0; done
  else
    run "${ds}_s4_block8_cr8_dp0.02_n${NIMG_DP}" "${dp[@]}" --defense block_random --block-size 8 --cr 8 --wcc-lambda 0 --clip-norm "${CLIP[${ds}_s4_b8]}"
  fi
done; done
wait
"$PY" "$REPRO_DIR/aggregate_unsplit.py" --results "$RUN_DIR/results" --out "$OUT/$TAB.csv"
