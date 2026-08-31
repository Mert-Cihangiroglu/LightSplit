#!/usr/bin/env bash
# block_random SDAR privacy sweep (locality dial): CR x block_size x seeds, lane-distributed.
# Mirrors run_sdar_grid.sh. Resume-guarded; logs reseed_logs/<setting>_l<level>_blockrandom_*.
# block_size is the dial: 1 = per-position (least mixing), 8 = global == LightSplit-F.
set -uo pipefail
cd "$(dirname "$0")"
SETTING="${SETTING:-vsl}"
LEVEL="${LEVEL:-7}"
CRS="${CRS:-8 16 32}"
BLOCKS="${BLOCKS:-1 2 4 8}"
RUNS="${RUNS:-0 1 2}"
WCCS="${WCCS:-0}"                # WCC lambdas; 0 = projection-only (back-compat default)
GPUS="${GPUS:-2 3}"
LANES_PER_GPU="${LANES_PER_GPU:-3}"
DRY="${DRY:-0}"
mkdir -p reseed_logs

jobs=()
for cr in $CRS; do for b in $BLOCKS; do for r in $RUNS; do for w in $WCCS; do
  wtag=""; [ "$w" != "0" ] && wtag="_wcc${w}"
  log="reseed_logs/${SETTING}_l${LEVEL}_blockrandom_cr${cr}_b${b}${wtag}_run${r}.log"
  grep -q "Average SSIM" "$log" 2>/dev/null && continue       # already done
  jobs+=("${cr}:${b}:${r}:${w}")
done; done; done; done

lanes=(); for g in $GPUS; do for ((i=0; i<LANES_PER_GPU; i++)); do lanes+=("$g"); done; done
nlanes=${#lanes[@]}
echo "setting=$SETTING level=l$LEVEL   jobs to run: ${#jobs[@]}   lanes: ${lanes[*]} (n=$nlanes)"

declare -A lj
for idx in "${!jobs[@]}"; do l=$((idx % nlanes)); lj[$l]+="${jobs[$idx]} "; done

for l in "${!lanes[@]}"; do
  g=${lanes[$l]}; jl="${lj[$l]:-}"; [ -z "$jl" ] && continue
  sess="brgrid_${SETTING}_n${l}g${g}"
  echo "  $sess  gpu$g  <- $jl"
  [ "$DRY" = 1 ] && continue
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new-session -d -s "$sess" -c "$PWD" \
    "SETTING=$SETTING LEVEL=$LEVEL GPU=$g JOBS='$jl' ./run_block_random_reseed.sh 2>&1 | tee reseed_logs/lane_${sess}.log"
done
[ "$DRY" = 1 ] && echo "(dry run: nothing launched)"
