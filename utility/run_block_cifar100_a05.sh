#!/usr/bin/env bash
# FILL: block-methods utility on cifar100 at the PAPER's canonical non-IID alpha=0.5
# (the appendix table uses niid0.5 for cifar100; the main grid ran niid0.1, so the
# cifar100 non-IID panel had no block lines). 4 heads x block{1,2,4} x 2 modes x CR{8,16,32}.
# One --sweep process per (head,block) => 6 rows (2 modes x 3 CR x 1 alpha).
# DISTINCT CSVs (suffix _a05) so this never collides with the alpha=0.1 grid's files
# or trips their resume guard. Resume-guarded; co-locates. Detach in tmux 'bgq05';
# each cell -> tmux bg05_<head>_b<B>.
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY=python
GPUS=(${GPUS:-0 1 2 3})
PER_GPU=${PER_GPU:-2}
DS=cifar100
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH})
BLOCKS=(${BLOCKS:-1 2 4})
MODES="${MODES:-block_random_ll block_random}"
CRS="8 16 32"
ALPHA="0.5"                                        # the paper's cifar100 non-IID
NC=10
EXPECT_ROWS=6                                      # 2 modes x 3 CR x 1 alpha
mkdir -p "$ML/logs" "$ML/results/csv"
log(){ echo "[bgq05 $(date '+%F %T')] $*"; }

head_flags(){ case "$1" in
  L1H_SCH) echo "" ;;
  L1H_PCH) echo "--client-heads" ;;
  L2H_SCH) echo "--deep-head" ;;
  L2H_PCH) echo "--deep-head --client-heads" ;;
  *) echo "BADHEAD" ;; esac; }

JOBS=()
for H in "${HEADS[@]}"; do for B in "${BLOCKS[@]}"; do JOBS+=("$H:$B"); done; done

sess_of(){ echo "bg05_${1}_b${2}"; }
sum_of(){  echo "results/csv/util2k_${DS}_${1}_b${2}_a05_summary.csv"; }
rows_of(){ [ -f "$1" ] && awk 'END{print NR-1}' "$1" 2>/dev/null || echo 0; }
is_done(){ [ "$(rows_of "$(sum_of "$1" "$2")")" -ge "$EXPECT_ROWS" ]; }

lane_free(){ # gpu free if fewer than PER_GPU bg05_* sessions pinned to it
  local g=$1 s pid cmd n=0
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^bg05_'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && n=$((n+1))
  done
  [ "$n" -lt "$PER_GPU" ]
}

launch(){ # H B gpu
  local H=$1 B=$2 g=$3 sess hf
  sess=$(sess_of "$H" "$B"); hf=$(head_flags "$H")
  log "GPU $g <- $DS $H b$B alpha=$ALPHA  (tmux $sess)"
  tmux new -s "$sess" -d "cd $ML && $PY -u train_split.py --sweep --dataset $DS \
    --modes $MODES --block-size $B --compression-ratio-list $CRS --alpha-list $ALPHA \
    --num-clients $NC --epochs 2000 --patience 100 --gpu $g $hf --no-viz --resume \
    --summary-csv $(sum_of "$H" "$B") \
    --epoch-csv results/csv/util2k_${DS}_${H}_b${B}_a05_epochs.csv 2>&1 | tee logs/${sess}.log"
}

log "cifar100 a=0.5 FILL: ${#JOBS[@]} jobs (x${EXPECT_ROWS} rows), lanes=[${GPUS[*]}]x${PER_GPU}, modes=[$MODES]"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r H B <<< "$j"
    tmux has-session -t "$(sess_of "$H" "$B")" 2>/dev/null && continue
    is_done "$H" "$B" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all jobs done/running; exiting."; break; }
  free_gpu=""
  for g in "${GPUS[@]}"; do lane_free "$g" && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r H B <<< "${pending[0]}"
    launch "$H" "$B" "$free_gpu"
    sleep 20
  else
    sleep 30
  fi
done
log "cifar100 a=0.5 FILL dispatcher exiting."
