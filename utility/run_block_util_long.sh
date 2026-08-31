#!/usr/bin/env bash
# LONG (2000ep + patience100) block_random WCC *utility* sweep, CIFAR-10, the utility harness.
# Builds the converged-accuracy side of the privacy/utility curve, matching the
# fixed_random wcc_check 2000ep runs already in flight, so every mode is on the same
# training budget (no more 100ep-underfit block numbers).
#   block  in {1,2,4}                          (b8 == fixed_random already covered by wcc_check)
#   lambda in {0,0.001,0.003,0.01,0.03,0.1}    (usable WCC regime; pairs with the known-R recon cells)
#   CR 8  (matches the privileged known-R / id_rank analysis, all CR8/k512)
# Resume-guarded: a cell whose summary.csv is non-empty (= finished) is skipped; a cell
# whose tmux session exists (= running) is skipped. Co-locates 1 job/GPU across lanes
# (each run ~4 GB / 48 GB card). Override lanes:  GPUS="0 1" bash run_block_util_long.sh
# Detach: launch this whole script inside tmux 'lbq'; each cell gets its own tmux lb_b{B}_l{L}.
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY=python      # pytorchEnv (LightSplit env does not exist)
GPUS=(${GPUS:-0 1 2 3})                            # co-locate 1 lb job per card across all 4
mkdir -p "$ML/logs" "$ML/results/csv"
log(){ echo "[lbq $(date '+%F %T')] $*"; }

BLOCKS=(1 2 4)
LAMBDAS=(0 0.001 0.003 0.01 0.03 0.1)
JOBS=()
for B in "${BLOCKS[@]}"; do for L in "${LAMBDAS[@]}"; do JOBS+=("$B:$L"); done; done

sess_of(){ local l="${2//./_}"; echo "lb_b${1}_l${l}"; }          # tmux rewrites '.'->'_'; match it
tag_of(){  echo "utillong_block_random_b${1}_cifar10_L2H_SCH_w${2}"; }
sum_of(){  echo "results/csv/$(tag_of "$1" "$2")_summary.csv"; }
is_done(){ [ -s "$(sum_of "$1" "$2")" ]; }                        # non-empty summary = complete cell

lane_free(){ # gpu -> 0 (free) if no running lb_* session is pinned to it
  local g=$1 s pid cmd
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^lb_b'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && return 1
  done
  return 0
}

launch(){ # B L gpu
  local B=$1 L=$2 g=$3 sess tag
  sess=$(sess_of "$B" "$L"); tag=$(tag_of "$B" "$L")
  log "GPU $g <- block=$B lambda=$L  (tmux $sess, csv $tag)"
  tmux new -s "$sess" -d "cd $ML && $PY -u train_split.py --sweep \
    --modes block_random --block-size $B --compression-ratio-list 8 \
    --alpha-list 10000000 --num-clients 10 --deep-head --wcc-lambda $L \
    --dataset cifar10 --epochs 2000 --patience 100 --gpu $g --resume --no-viz \
    --summary-csv results/csv/${tag}_summary.csv \
    --epoch-csv results/csv/${tag}_epochs.csv 2>&1 | tee logs/${sess}.log"
}

log "block util LONG sweep: ${#JOBS[@]} cells, lanes=[${GPUS[*]}]  (2000ep+pat100, CR8, resume-guarded)"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r B L <<< "$j"
    tmux has-session -t "$(sess_of "$B" "$L")" 2>/dev/null && continue
    is_done "$B" "$L" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all cells done/running; exiting."; break; }

  free_gpu=""
  for g in "${GPUS[@]}"; do lane_free "$g" && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r B L <<< "${pending[0]}"
    launch "$B" "$L" "$free_gpu"
    sleep 20
  else
    sleep 30
  fi
done
log "block util LONG dispatcher exiting."
