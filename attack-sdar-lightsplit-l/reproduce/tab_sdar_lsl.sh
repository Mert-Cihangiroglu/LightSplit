#!/usr/bin/env bash
# Table tab:sdar_lsl: SDAR against LightSplit-L (fixed_random_ll), CIFAR-10 USL,
# level 7 (4096-d cut), lambda_WCC = 0, CR {8, 16, 32}.
# The paper's rows are means over independent attacker runs (3 at CR8, 6 at CR16/32;
# SDAR runs are not seeded). Each run is a full SDAR training (hours on one GPU).
# The Raw and LightSplit-F reference rows of the table come from attack-sdar's
# tab_sdar_all_split_points.sh.
# Output: $OUT logs + reproduce/results/tab_sdar_lsl.csv
set -uo pipefail
source "$(dirname "$0")/common.sh"
RUNS8="${RUNS8:-0 1 2}"; RUNS="${RUNS:-0 1 2 3 4 5}"
SPECS=""
for r in $RUNS8; do SPECS+="usl_l7_frll_cr8_run${r}|usl resnet cifar10 7 $r|--bottleneck_mode fixed_random_ll --compression_ratio 8
"; done
for cr in 16 32; do for r in $RUNS; do SPECS+="usl_l7_frll_cr${cr}_run${r}|usl resnet cifar10 7 $r|--bottleneck_mode fixed_random_ll --compression_ratio $cr
"; done; done
printf '%s' "$SPECS" | run_specs
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/score_lsl.py" --logs "$OUT" --scripts "$CODE_DIR/scripts" --out "${CSV_OUT:-$REPRO_DIR/results}/tab_sdar_lsl.csv"
