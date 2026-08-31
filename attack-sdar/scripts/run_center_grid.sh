#!/usr/bin/env bash
# SDAR privacy DISPATCHER for CENTER LOSS -- CR x center_lambda x seeds across GPU lanes.
# Mirrors run_sdar_grid.sh. Records SSIM per run; accuracy only flags healthy vs collapsed
# victims (the paper's accuracy is the utility harness's 10-client setup). The WCC privacy cells
# at the matched config already exist (run_sdar_grid.sh WCCS=...), so this fills the center
# arm only. Resume-guarded; distributes across tmux lanes.
# Detach: tmux new -s x8sdar -d "GPUS='0 1' ./run_center_grid.sh"
set -uo pipefail
cd "$(dirname "$0")"
CRS="${CRS:-8}"
CENTERS="${CENTERS:-0.01 0.1}"
RUNS="${RUNS:-1 2 3}"
GPUS="${GPUS:-0 1}"
LANES_PER_GPU="${LANES_PER_GPU:-1}"
LEVEL="${LEVEL:-7}"                        # 7 = deep 4096-d cut == deployed model
DRY="${DRY:-0}"
mkdir -p reseed_logs

jobs=()
for cr in $CRS; do for c in $CENTERS; do for r in $RUNS; do
  log="reseed_logs/usl_l${LEVEL}_cr${cr}_center${c}_run${r}.log"
  grep -q "Average SSIM" "$log" 2>/dev/null && continue       # already done
  jobs+=("${cr}:${c}:${r}")
done; done; done

lanes=(); for g in $GPUS; do for ((i=0; i<LANES_PER_GPU; i++)); do lanes+=("$g"); done; done
nlanes=${#lanes[@]}
echo "level=l$LEVEL   jobs to run: ${#jobs[@]}   lanes: ${lanes[*]} (n=$nlanes)"
[ "${#jobs[@]}" -eq 0 ] && { echo "nothing to do"; exit 0; }

declare -A lj
for idx in "${!jobs[@]}"; do l=$((idx % nlanes)); lj[$l]+="${jobs[$idx]} "; done

for l in "${!lanes[@]}"; do
  g=${lanes[$l]}; jl="${lj[$l]:-}"; [ -z "$jl" ] && continue
  sess="centergrid_l${LEVEL}_n${l}g${g}"
  echo "  $sess  gpu$g  <- $jl"
  [ "$DRY" = 1 ] && continue
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new-session -d -s "$sess" -c "$PWD" \
    "LEVEL=$LEVEL GPU=$g JOBS='$jl' ./run_center_reseed.sh 2>&1 | tee reseed_logs/lane_${sess}.log"
done
[ "$DRY" = 1 ] && echo "(dry run: nothing launched)"
