#!/usr/bin/env bash
# LONG (2000ep + patience100) block_random WCC *utility* sweep at extra compression ratios.
# Companion to run_block_util_long.sh (which did CR8). Adds CR16/CR32 so the utility grid
# spans the whole compression axis, every mode on the same 2000ep budget.
#   block  in {1,2,4}
#   lambda in {0,0.001,0.003,0.01,0.03,0.1}
#   CR     in {16,32}   (k = 4096/CR -> 256, 128; all block sizes divide cleanly)
# Tag includes cr (CR8 files from the sibling script have NO cr in the name = implicitly CR8).
# Resume-guarded (non-empty summary.csv = done; live tmux session = running). Co-locates
# 1 job/GPU; lane_free counts only THIS sweep's lc_* sessions. Override: CRS="16" GPUS="2 3" ...
# Detach: launch inside tmux 'lcq'; each cell -> tmux lc{CR}_b{B}_l{L}.
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY=python
GPUS=(${GPUS:-0 1 2 3})
CRS=(${CRS:-16 32})
BLOCKS=(1 2 4)
LAMBDAS=(0 0.001 0.003 0.01 0.03 0.1)
mkdir -p "$ML/logs" "$ML/results/csv"
log(){ echo "[lcq $(date '+%F %T')] $*"; }

# front-load lambda=0 across all (block,CR) so the no-defense utility-vs-CR grid lands first
JOBS=()
for L in "${LAMBDAS[@]}"; do for B in "${BLOCKS[@]}"; do for CR in "${CRS[@]}"; do
  JOBS+=("$B:$CR:$L"); done; done; done

sess_of(){ local l="${3//./_}"; echo "lc${2}_b${1}_l${l}"; }                 # lc16_b1_l0 ...
tag_of(){  echo "utillong_block_random_b${1}_cr${2}_cifar10_L2H_SCH_w${3}"; }
sum_of(){  echo "results/csv/$(tag_of "$1" "$2" "$3")_summary.csv"; }
is_done(){ [ -s "$(sum_of "$1" "$2" "$3")" ]; }

lane_free(){ # gpu -> 0 (free) if no running lc_* session is pinned to it
  local g=$1 s pid cmd
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^lc[0-9]'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && return 1
  done
  return 0
}

launch(){ # B CR L gpu
  local B=$1 CR=$2 L=$3 g=$4 sess tag
  sess=$(sess_of "$B" "$CR" "$L"); tag=$(tag_of "$B" "$CR" "$L")
  log "GPU $g <- block=$B CR=$CR lambda=$L  (tmux $sess)"
  tmux new -s "$sess" -d "cd $ML && $PY -u train_split.py --sweep \
    --modes block_random --block-size $B --compression-ratio-list $CR \
    --alpha-list 10000000 --num-clients 10 --deep-head --wcc-lambda $L \
    --dataset cifar10 --epochs 2000 --patience 100 --gpu $g --resume --no-viz \
    --summary-csv results/csv/${tag}_summary.csv \
    --epoch-csv results/csv/${tag}_epochs.csv 2>&1 | tee logs/${sess}.log"
}

log "block util LONG sweep: ${#JOBS[@]} cells (CR=[${CRS[*]}]), lanes=[${GPUS[*]}], 2000ep+pat100, resume-guarded"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r B CR L <<< "$j"
    tmux has-session -t "$(sess_of "$B" "$CR" "$L")" 2>/dev/null && continue
    is_done "$B" "$CR" "$L" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all cells done/running; exiting."; break; }

  free_gpu=""
  for g in "${GPUS[@]}"; do lane_free "$g" && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r B CR L <<< "${pending[0]}"
    launch "$B" "$CR" "$L" "$free_gpu"
    sleep 20
  else
    sleep 30
  fi
done
log "block util LONG (CR) dispatcher exiting."
