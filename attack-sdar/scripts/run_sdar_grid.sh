#!/usr/bin/env bash
# Full multi-seed SDAR privacy sweep: CR x WCC x seeds. Records SSIM + clean task accuracy
# per run (the latter only to identify healthy vs collapsed victims; the paper's accuracy
# comes from the utility harness's 10-client setup, not from SDAR's one client).
# run1 already exists in the original blackbox_* logs; this runs seeds 2..N into reseed_logs/.
# Resume-guarded (skips a run whose log already has the SSIM line). Distributes across lanes.
set -uo pipefail
cd "$(dirname "$0")"
CRS="${CRS:-8 16 32}"
WCCS="${WCCS:-0 0.01 0.1}"
RUNS="${RUNS:-2 3 4 5}"
GPUS="${GPUS:-2 3}"
LANES_PER_GPU="${LANES_PER_GPU:-3}"
LEVEL="${LEVEL:-4}"                        # SDAR split level (7 = deep 4096-d cut == deployed model)
DRY="${DRY:-0}"
mkdir -p reseed_logs

jobs=()
for cr in $CRS; do for w in $WCCS; do for r in $RUNS; do
  log="reseed_logs/usl_l${LEVEL}_cr${cr}_wcc${w}_run${r}.log"
  grep -q "Average SSIM" "$log" 2>/dev/null && continue       # already done
  jobs+=("${cr}:${w}:${r}")
done; done; done

lanes=(); for g in $GPUS; do for ((i=0; i<LANES_PER_GPU; i++)); do lanes+=("$g"); done; done
nlanes=${#lanes[@]}
echo "level=l$LEVEL   jobs to run: ${#jobs[@]}   lanes: ${lanes[*]} (n=$nlanes)"

declare -A lj
for idx in "${!jobs[@]}"; do l=$((idx % nlanes)); lj[$l]+="${jobs[$idx]} "; done

for l in "${!lanes[@]}"; do
  g=${lanes[$l]}; jl="${lj[$l]:-}"; [ -z "$jl" ] && continue
  sess="sdargrid_l${LEVEL}_n${l}g${g}"
  echo "  $sess  gpu$g  <- $jl"
  [ "$DRY" = 1 ] && continue
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new-session -d -s "$sess" -c "$PWD" \
    "LEVEL=$LEVEL GPU=$g JOBS='$jl' ./run_wcc_reseed.sh 2>&1 | tee reseed_logs/lane_${sess}.log"
done
[ "$DRY" = 1 ] && echo "(dry run: nothing launched)"
