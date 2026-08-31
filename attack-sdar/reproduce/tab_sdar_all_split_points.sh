#!/usr/bin/env bash
# Table tab:final-sdar-usl-all-split-points (merged USL+VSL SDAR sweep on CIFAR-10),
# plus its two derived tables: tab:sdar-main-compact (USL level-4 CR8 slice) and
# tab:dp_privacy (the DP SSIM cells).
# Grid per protocol {usl, vsl} x level {7 = 4096-d cut, 4 = 8192-d cut}:
#   raw, learned_1x1 CR{8,16,32}, fixed_random CR{8,16,32} x lambda_WCC {0, 0.01, 0.1}: 1 run each
#   topk_sparse CR{8,16,32}: 3 runs (the attacker sometimes fails to train; the
#     aggregator averages runs whose attacker converged, label acc >= 0.40)
#   raw + DP and fixed_random CR8 + DP (sigma 0.02): 3 runs each (6 for USL level 7
#     fixed_random CR8 + DP), averaged. Clip = median payload l2-norm measured by
#     scripts/measure_dp_norms.py: level 7 raw 62.75 / CR8 20.89; level 4 raw 88.46 / CR8 30.87.
# 152 attacker runs, each a full SDAR training (hours on one GPU). SDAR does not seed its
# runs (each run id is an independent trajectory), so re-run cells scatter around the
# printed values; the aggregation protocol is what the table fixes.
# SETTINGS/LEVELS/MODES restrict the grid. Output: $OUT plus three CSVs via the aggregator.
set -uo pipefail
source "$(dirname "$0")/common.sh"
SETTINGS="${SETTINGS:-usl vsl}"
LEVELS="${LEVELS:-7 4}"
MODES="${MODES:-raw learned_1x1 fixed_random topk_sparse dp}"
declare -A CLIP_RAW=( [7]=62.75 [4]=88.46 )
declare -A CLIP_CR8=( [7]=20.89 [4]=30.87 )

SPECS=""
for s in $SETTINGS; do for L in $LEVELS; do
  for mode in $MODES; do case "$mode" in
    raw) SPECS+="${s}_l${L}_raw_run1|$s resnet cifar10 $L 1|
";;
    learned_1x1) for cr in 8 16 32; do
      SPECS+="${s}_l${L}_learned1x1_cr${cr}_run1|$s resnet cifar10 $L 1|--bottleneck_mode learned_1x1 --compression_ratio $cr
"; done;;
    fixed_random) for cr in 8 16 32; do for w in 0 0.01 0.1; do
      SPECS+="${s}_l${L}_fixedrandom_cr${cr}_wcc${w}_run1|$s resnet cifar10 $L 1|--bottleneck_mode fixed_random --compression_ratio $cr --wcc_lambda $w
"; done; done;;
    topk_sparse) for cr in 8 16 32; do for r in 0 1 2; do
      SPECS+="${s}_l${L}_topksparse_cr${cr}_run${r}|$s resnet cifar10 $L $r|--bottleneck_mode topk_sparse --compression_ratio $cr
"; done; done;;
    dp) for r in 1 2 3; do
      SPECS+="${s}_l${L}_rawdp_run${r}|$s resnet cifar10 $L $r|--dp_clip_norm ${CLIP_RAW[$L]} --dp_sigma 0.02
";
      SPECS+="${s}_l${L}_cr8dp_run${r}|$s resnet cifar10 $L $r|--bottleneck_mode fixed_random --compression_ratio 8 --dp_clip_norm ${CLIP_CR8[$L]} --dp_sigma 0.02
"; done
      if [ "$s" = usl ] && [ "$L" = 7 ]; then for r in 4 5 6; do
        SPECS+="usl_l7_cr8dp_run${r}|usl resnet cifar10 7 $r|--bottleneck_mode fixed_random --compression_ratio 8 --dp_clip_norm ${CLIP_CR8[7]} --dp_sigma 0.02
"; done; fi;;
  esac; done
done; done

# one extra cell for the consolidated privacy table: LS-F lambda=1e-3 at the USL 8192-d cut
case " $SETTINGS " in *" usl "*) case " $LEVELS " in *" 4 "*) case " $MODES " in *fixed_random*)
  SPECS+="usl_l4_fixedrandom_cr8_wcc0.001_run1|usl resnet cifar10 4 1|--bottleneck_mode fixed_random --compression_ratio 8 --wcc_lambda 0.001
";; esac;; esac;; esac

printf '%s' "$SPECS" | run_specs
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_sdar_tables.py" --logs "$OUT" --scripts "$CODE_DIR/scripts" --out-dir "${CSV_OUT:-$REPRO_DIR/results}"
