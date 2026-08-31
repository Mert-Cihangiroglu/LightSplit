#!/usr/bin/env bash
# Table tab:full_sweep_baseline_appendix (U-shaped SL utility sweep) and the utility
# columns of tab:main-compact.
# Grid: {Raw, LightSplit-F, LightSplit-L, Learned 1x1, topk_sparse}
#       x {CIFAR-10, CIFAR-100, FashionMNIST, GTSRB} x {non-IID, IID}
#       x {L1H, L2H} x {SCH, PCH} x CR {8, 16, 32}   -> 416 runs, 10 clients each.
# Epoch budgets reproduce the paper's runs per cell:
#   2000 epochs with early stopping (patience 200): FashionMNIST and GTSRB (all methods
#     but topk_sparse), LightSplit-L everywhere, and the CIFAR-100 LightSplit-F PCH cells;
#   200 epochs: the remaining CIFAR-100 SCH cells; 100 epochs: CIFAR-10 and the
#     CIFAR-100 PCH Raw / Learned 1x1 cells; topk_sparse: 100 epochs everywhere.
# JOBS=N trains N cells concurrently (default 1; they share the GPU).
# SMOKE=1 runs a 4-run subset at 2 epochs.
# Output: $OUT/tab_full_sweep.csv and $OUT/tab_full_sweep_main_utility.csv
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=tab_full_sweep
JOBS="${JOBS:-1}"
DATASETS="${DATASETS:-cifar10 cifar100 fmnist gtsrb}"
MODES="${MODES:-raw fixed_random fixed_random_ll learned_1x1 topk_sparse}"
CRS="${CRS:-8 16 32}"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then DATASETS=cifar10; MODES="raw fixed_random fixed_random_ll topk_sparse"; CRS=8; fi

budget() { # ds mode head -> "epochs patience"
  local ds=$1 mode=$2 head=$3
  if [ "$SMOKE" = 1 ]; then echo "2 0"; return; fi
  case "$mode" in
    topk_sparse) echo "100 0"; return ;;
    fixed_random_ll) echo "2000 200"; return ;;
  esac
  case "$ds" in
    fmnist|gtsrb) echo "2000 200" ;;
    cifar10) echo "100 0" ;;
    cifar100)
      if [ "$mode" = fixed_random ] && [ "$head" = PCH ]; then echo "2000 200"
      elif [ "$head" = PCH ]; then echo "100 0"
      else echo "200 0"; fi ;;
  esac
}
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( run "$@" ) & }

for ds in $DATASETS; do
  case "$ds" in cifar100) NA=0.5 ;; *) NA=0.1 ;; esac
  for dist in noniid iid; do
    if [ "$dist" = noniid ]; then alpha=$NA; else alpha=10000000; fi
    [ "$SMOKE" = 1 ] && [ "$dist" = noniid ] && continue
    for depth in L1H L2H; do for headc in SCH PCH; do
      if [ "$SMOKE" = 1 ] && { [ "$depth" != L2H ] || [ "$headc" != SCH ]; }; then continue; fi
      flags=(); [ "$depth" = L2H ] && flags+=(--deep-head); [ "$headc" = PCH ] && flags+=(--client-heads)
      for mode in $MODES; do
        read -r ep pat <<< "$(budget "$ds" "$mode" "$headc")"
        base=(--dataset "$ds" --mode "$mode" --num-clients 10 --alpha "$alpha" --epochs "$ep")
        [ "$pat" != 0 ] && base+=(--patience "$pat")
        if [ "$mode" = raw ]; then
          runq "$FIG" "${ds}_${dist}_${depth}_${headc}_raw" "${base[@]}" ${flags[@]+"${flags[@]}"}
        else
          for cr in $CRS; do
            runq "$FIG" "${ds}_${dist}_${depth}_${headc}_${mode}_cr${cr}" "${base[@]}" --compression-ratio "$cr" ${flags[@]+"${flags[@]}"}
          done
        fi
      done
    done; done
  done
done
wait || true
"$PY" "$REPRO_DIR/aggregate_full_sweep.py" --results "$OUT/$FIG" --out "$OUT/tab_full_sweep.csv"
