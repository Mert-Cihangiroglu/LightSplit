#!/usr/bin/env bash
# Table tab:wcc_ablation — accuracy under WCC across head configs, for the
# global projection and the block-local one (b=2). CIFAR-10, IID, CR 8,
# 10 clients, 2000 epochs / patience 100.
# Runs: {LightSplit-F, LightSplit-L} x {global, b=2} x {L1H,L2H}x{SCH,PCH}
#       x lambda {0, 1e-3, 1e-2, 1e-1} = 64 runs, plus 4 raw runs that anchor
# the b=2 rows (the global rows read against the full utility sweep's Raw
# cells: run tab_full_sweep.sh first). A few GPU-days. SMOKE=1: two cells.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=wcc_ablation
JOBS="${JOBS:-1}"
HEADS="${HEADS:-L1H_SCH L2H_SCH L1H_PCH L2H_PCH}"
LAMS="${LAMS:-0 0.001 0.01 0.1}"
PROJS="${PROJS:-global b2}"
EP=2000; PAT=100
if [ "${SMOKE:-0}" = 1 ]; then HEADS=L2H_SCH; LAMS=0; PROJS=b2; EP=2; PAT=0; fi
mkdir -p "$OUT/$FIG"
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( run "$FIG" "$@" ) & }

for H in $HEADS; do
  flags=(); [ "${H%%_*}" = L2H ] && flags+=(--deep-head); [ "${H##*_}" = PCH ] && flags+=(--client-heads)
  base=(--dataset cifar10 --num-clients 10 --alpha 10000000 --epochs "$EP")
  [ "$PAT" != 0 ] && base+=(--patience "$PAT")
  runq "raw_${H}" "${base[@]}" --mode raw ${flags[@]+"${flags[@]}"}
  for proj in $PROJS; do
    case "$proj" in
      global) mf=fixed_random; ml=fixed_random_ll; extra=() ;;
      b2)     mf=block_random; ml=block_random_ll; extra=(--block-size 2) ;;
    esac
    for w in $LAMS; do
      runq "${proj}_F_${H}_w${w}" "${base[@]}" --mode "$mf" --compression-ratio 8 \
        --wcc-lambda "$w" ${extra[@]+"${extra[@]}"} ${flags[@]+"${flags[@]}"}
      runq "${proj}_L_${H}_w${w}" "${base[@]}" --mode "$ml" --compression-ratio 8 \
        --wcc-lambda "$w" ${extra[@]+"${extra[@]}"} ${flags[@]+"${flags[@]}"}
    done
  done
done
wait
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_wcc_ablation.py" \
  --results "$OUT/$FIG" --fullsweep "$OUT/tab_full_sweep" --out "$OUT/tab_wcc_ablation.csv"
