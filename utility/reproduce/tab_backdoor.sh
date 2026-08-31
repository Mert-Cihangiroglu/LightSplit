#!/usr/bin/env bash
# Table tab:bd-ours — label-free backdoor detection F1 on the backbone output.
# Baseline: IID, n=10, mal 0.1, poison 0.3, CR 16, seeds {42,123,2024}, 50
# epochs. Ablations change one parameter each (alpha, n, mal ratio, poison
# rate), single runs at 30 epochs, seed 42. fixed_random plus the
# fixed_random_ll twin per setting: 30 runs, roughly a GPU-day.
# SMOKE=1 runs one baseline seed at 2 epochs.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=backdoor
JOBS="${JOBS:-1}"
MODES="${MODES:-fixed_random fixed_random_ll}"
SEEDS="${SEEDS:-42 123 2024}"
EPB=50; EPA=30
if [ "${SMOKE:-0}" = 1 ]; then MODES=fixed_random; SEEDS=42; EPB=2; EPA=2; fi
mkdir -p "$OUT/$FIG"
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( bd "$@" ) & }
bd() { # epochs n mal pr alpha seed mode
  local dir="$OUT/$FIG/cifar10_${7}_c${2}_mal${3}_pr${4}_k$((4096/16*0+256))_seed${6}_a${5}"
  # run_backdoor.py names its own output dir; skip when its stats exist
  local name="cifar10_${7}_c${2}_mal${3}_pr${4}_k256_seed${6}_a${5}"
  [ -s "$OUT/$FIG/$name/detection_stats.csv" ] && { echo "[skip] $name"; return; }
  echo "[$(date +%H:%M:%S)] $name"
  "$PY" "$CODE_DIR/run_backdoor.py" --dataset cifar10 --mode "$7" --k 16 \
    --epochs "$1" --num-clients "$2" --malicious-ratio "$3" --poison-rate "$4" \
    --alpha "$5" --seed "$6" --target-class 0 --trigger-size 3 \
    --batch-size 128 --num-workers 2 --out-dir "$OUT/$FIG" --gpu 0
}
for mode in $MODES; do
  for seed in $SEEDS; do runq "$EPB" 10 0.1 0.3 10000000 "$seed" "$mode"; done
  if [ "${SMOKE:-0}" != 1 ]; then
    for al in 0.1 1 10; do runq "$EPA" 10 0.1 0.3 "$al" 42 "$mode"; done
    for n in 5 20 50;   do runq "$EPA" "$n" 0.1 0.3 10000000 42 "$mode"; done
    for m in 0.2 0.3;   do runq "$EPA" 10 "$m" 0.3 10000000 42 "$mode"; done
    for p in 0.1 0.2 0.5 0.8; do runq "$EPA" 10 0.1 "$p" 10000000 42 "$mode"; done
  fi
done
wait
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_backdoor.py" --results "$OUT/$FIG" --out "$OUT/tab_backdoor.csv"
