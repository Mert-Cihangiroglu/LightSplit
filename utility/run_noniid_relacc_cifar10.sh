#!/usr/bin/env bash
# ============================================================================
# Non-IID CIFAR-10 "relative-to-raw" utility sweep.
# ----------------------------------------------------------------------------
# Deliverable: per head-config {L1H,L2H}x{SCH,PCH}, relative accuracy change vs
# the raw (no-defense) baseline, across CR {8,16,32}, for:
#     * block_random  block {1,2,4}      -- OUR method, 2000ep + patience100
#     * learned_1x1                      -- baseline,  100ep
#     * raw                              -- reference, 100ep
#     * topk_sparse                      -- baseline, REUSED from mb_cifar10_* (100ep); not re-run here
#
# Non-IID alpha = 0.1, 10 clients, resnet18.  WCC lambda = 0 (utility only).
# train_split --sweep iterates modes x CR x alpha internally, so:
#     * block_random: 1 process per (head,block) -> sweeps 3 CR  (3 rows)   x4 heads x3 blocks = 12 procs
#     * baselines   : 1 process per head, --modes "raw learned_1x1" -> raw(1) + learned(3 CR) = 4 rows  x4 = 4 procs
# Resume-safe: train_split --resume skips done cells; dispatcher skips full summaries / live sessions.
# Co-locates CAP jobs/GPU.  Detach: run dispatcher inside tmux 'rlq'; each cell -> tmux rl_<...>.
# ============================================================================
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY=python
GPUS=(${GPUS:-0 1})
CAP="${CAP:-2}"                       # max concurrent jobs per GPU
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH})
BLOCKS=(${BLOCKS:-1 2 4})
CRS="8 16 32"
ALPHA=0.1                             # non-IID
NC=10
mkdir -p "$ML/logs" "$ML/results/csv"
log(){ echo "[rlq $(date '+%F %T')] $*"; }

head_flags(){ case "$1" in
  L1H_SCH) echo "" ;;
  L1H_PCH) echo "--client-heads" ;;
  L2H_SCH) echo "--deep-head" ;;
  L2H_PCH) echo "--deep-head --client-heads" ;;
  *) echo "BADHEAD" ;; esac; }

# ---- job list: "fam:head:block" (block '-' for baselines) ----
JOBS=()
for H in "${HEADS[@]}"; do JOBS+=("base:$H:-"); done
for H in "${HEADS[@]}"; do for B in "${BLOCKS[@]}"; do JOBS+=("br:$H:$B"); done; done

sess_of(){  # fam head block
  if [ "$1" = base ]; then echo "rl_base_${2}"; else echo "rl_br_${2}_b${3}"; fi; }
sum_of(){
  if [ "$1" = base ]; then echo "results/csv/relacc100_baselines_cifar10_${2}_summary.csv";
  else echo "results/csv/relacc2k_block_random_b${3}_cifar10_${2}_summary.csv"; fi; }
epo_of(){
  if [ "$1" = base ]; then echo "results/csv/relacc100_baselines_cifar10_${2}_epochs.csv";
  else echo "results/csv/relacc2k_block_random_b${3}_cifar10_${2}_epochs.csv"; fi; }
expect_of(){ [ "$1" = base ] && echo 4 || echo 3; }   # base: raw+3CR ; br: 3CR
rows_of(){ [ -f "$1" ] && awk 'END{print NR-1}' "$1" 2>/dev/null || echo 0; }
is_done(){ local f; f="$(sum_of "$1" "$2" "$3")"; [ "$(rows_of "$f")" -ge "$(expect_of "$1")" ]; }

lane_count(){ # gpu -> number of running rl_* sessions pinned to it
  local g=$1 s pid cmd n=0
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^rl_'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && n=$((n+1))
  done
  echo "$n"; }

launch(){ # fam head block gpu
  local fam=$1 H=$2 B=$3 g=$4 sess hf cmd
  sess=$(sess_of "$fam" "$H" "$B"); hf=$(head_flags "$H")
  if [ "$fam" = base ]; then
    cmd="$PY -u train_split.py --sweep --dataset cifar10 --modes raw learned_1x1 \
      --compression-ratio-list $CRS --alpha-list $ALPHA --num-clients $NC \
      --epochs 100 --gpu $g $hf --no-viz --resume \
      --summary-csv $(sum_of "$fam" "$H" "$B") --epoch-csv $(epo_of "$fam" "$H" "$B")"
    log "GPU $g <- baselines(raw+learned_1x1,100ep) $H  (tmux $sess)"
  else
    cmd="$PY -u train_split.py --sweep --dataset cifar10 --modes block_random --block-size $B \
      --compression-ratio-list $CRS --alpha-list $ALPHA --num-clients $NC \
      --epochs 2000 --patience 100 --gpu $g $hf --no-viz --resume \
      --summary-csv $(sum_of "$fam" "$H" "$B") --epoch-csv $(epo_of "$fam" "$H" "$B")"
    log "GPU $g <- block_random b$B (2000ep) $H  (tmux $sess)"
  fi
  tmux new -s "$sess" -d "cd $ML && $cmd 2>&1 | tee logs/${sess}.log"; }

log "non-iid relacc cifar10: ${#JOBS[@]} jobs, lanes=[${GPUS[*]}] cap=$CAP/gpu, blocks=[${BLOCKS[*]}]"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r fam H B <<< "$j"
    tmux has-session -t "$(sess_of "$fam" "$H" "$B")" 2>/dev/null && continue
    is_done "$fam" "$H" "$B" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all jobs done/running; dispatcher exiting."; break; }

  free_gpu=""
  for g in "${GPUS[@]}"; do [ "$(lane_count "$g")" -lt "$CAP" ] && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r fam H B <<< "${pending[0]}"
    launch "$fam" "$H" "$B" "$free_gpu"
    sleep 15
  else
    sleep 30
  fi
done
log "non-iid relacc cifar10 dispatcher exiting (pending=0)."
