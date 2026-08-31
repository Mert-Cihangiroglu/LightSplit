#!/usr/bin/env bash
# LONG (2000ep + patience100) block-methods UTILITY grid across deployment configs.
# Axes: dataset x {iid,non-iid} x head{L1H,L2H}x{SCH,PCH} x CR{8,16,32} x block{1,2,4}
#       x server-liftback { block_random (transpose->backbone), block_random_ll (server MLP) }.
# WCC lambda = 0 (utility only), num_clients = 10.
# train_split --sweep iterates modes x CR x alpha internally, so ONE batched process per
# (dataset,head,block) emits 12 rows (2 modes x 3 CR x 2 dist). 44 processes total
# (48 - 4 gtsrb-b4; 14x14 not divisible by 4 -> BlockRandomBottleneck raises).
# Resume-guarded: 12 rows in summary = done; live tmux session = running. Co-locates
# 1 job/GPU. Stage via env: DATASETS= HEADS= BLOCKS= GPUS= MODES=.
# Detach: run inside tmux 'bgq'; each cell -> tmux bg_<dataset>_<head>_b<B>.
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY=python      # pytorchEnv (LightSplit env does not exist)
GPUS=(${GPUS:-0 1 2 3})
PER_GPU=${PER_GPU:-3}                              # co-locate up to N cells/GPU (small models, ~0.8GB each)
DATASETS=(${DATASETS:-cifar10 cifar100 gtsrb fmnist})
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH})
BLOCKS=(${BLOCKS:-1 2 4})
MODES="${MODES:-block_random_ll block_random}"     # ll first: validate the NEW mode early
CRS="8 16 32"
ALPHAS="10000000 0.1"                              # iid (1e7) + non-iid (0.1)
NC=10
EXPECT_ROWS=12                                     # 2 modes x 3 CR x 2 alpha
mkdir -p "$ML/logs" "$ML/results/csv"
log(){ echo "[bgq $(date '+%F %T')] $*"; }

head_flags(){ case "$1" in
  L1H_SCH) echo "" ;;
  L1H_PCH) echo "--client-heads" ;;
  L2H_SCH) echo "--deep-head" ;;
  L2H_PCH) echo "--deep-head --client-heads" ;;
  *) echo "BADHEAD" ;; esac; }

# build job list (skip gtsrb b4)
JOBS=()
for D in "${DATASETS[@]}"; do for H in "${HEADS[@]}"; do for B in "${BLOCKS[@]}"; do
  [ "$D" = gtsrb ] && [ "$B" = 4 ] && continue
  JOBS+=("$D:$H:$B")
done; done; done

sess_of(){ echo "bg_${1}_${2}_b${3}"; }
sum_of(){  echo "results/csv/util2k_${1}_${2}_b${3}_summary.csv"; }
rows_of(){ [ -f "$1" ] && awk 'END{print NR-1}' "$1" 2>/dev/null || echo 0; }   # minus header
is_done(){ local f; f="$(sum_of "$1" "$2" "$3")"; [ "$(rows_of "$f")" -ge "$EXPECT_ROWS" ]; }

lane_free(){ # gpu -> 0 (free) if fewer than PER_GPU bg_* sessions are pinned to it
  local g=$1 s pid cmd n=0
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^bg_'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && n=$((n+1))
  done
  [ "$n" -lt "$PER_GPU" ]
}

launch(){ # D H B gpu
  local D=$1 H=$2 B=$3 g=$4 sess hf
  sess=$(sess_of "$D" "$H" "$B"); hf=$(head_flags "$H")
  log "GPU $g <- $D $H b$B  (tmux $sess)"
  tmux new -s "$sess" -d "cd $ML && $PY -u train_split.py --sweep --dataset $D \
    --modes $MODES --block-size $B --compression-ratio-list $CRS --alpha-list $ALPHAS \
    --num-clients $NC --epochs 2000 --patience 100 --gpu $g $hf --no-viz --resume \
    --summary-csv $(sum_of "$D" "$H" "$B") \
    --epoch-csv results/csv/util2k_${D}_${H}_b${B}_epochs.csv 2>&1 | tee logs/${sess}.log"
}

log "block-util GRID: ${#JOBS[@]} jobs (x${EXPECT_ROWS} cells each), lanes=[${GPUS[*]}], modes=[$MODES]"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r D H B <<< "$j"
    tmux has-session -t "$(sess_of "$D" "$H" "$B")" 2>/dev/null && continue
    is_done "$D" "$H" "$B" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all jobs done/running; exiting."; break; }

  free_gpu=""
  for g in "${GPUS[@]}"; do lane_free "$g" && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r D H B <<< "${pending[0]}"
    launch "$D" "$H" "$B" "$free_gpu"
    sleep 20
  else
    sleep 30
  fi
done
log "block-util GRID dispatcher exiting."
