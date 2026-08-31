#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WCC x UTILITY ladder for the MISSING head configs (extends the L2H_SCH-only
# WCC dial to the other three heads). cifar10, IID, 10 clients, 100 ep.
#   heads  : L1H_SCH L1H_PCH L2H_PCH        (L2H_SCH already done: utilwcc/wccsens)
#   modes  : block_random (b2, the sweet spot) + fixed_random (the dial baseline)
#   CR     : 8 16 32
#   lambda : 0 0.001 0.003 0.01 0.03 0.1
# One --sweep call per (head, lambda) emits 6 rows (2 modes x 3 CR). Separate CSV
# per lambda because train_split's --sweep resume key omits wcc_lambda (a shared
# CSV would falsely skip cells) -- same convention as utilwcc_*/wccsens_*.
# Resume-guarded (6 rows == done; live tmux session == running). Co-locates.
#   GPUS="0 1 2 3" PER_GPU=2 bash run_wcc_heads_grid.sh   (detach in tmux 'wccq')
# ---------------------------------------------------------------------------
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ML"
PY=python
GPUS=(${GPUS:-0 1 2 3})
PER_GPU=${PER_GPU:-2}
DS=cifar10
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_PCH})
LAMBDAS=(${LAMBDAS:-0 0.001 0.003 0.01 0.03 0.1})
MODES="block_random fixed_random"
BLOCK=2                                   # block_random operating point (sweet spot)
CRS="8 16 32"
ALPHA=10000000                            # IID
EPOCHS=${EPOCHS:-100}
NC=10
EXPECT_ROWS=6                             # 2 modes x 3 CR
mkdir -p logs results/csv
log(){ echo "[wccq $(date '+%F %T')] $*"; }

head_flags(){ case "$1" in
  L1H_SCH) echo "" ;;
  L1H_PCH) echo "--client-heads" ;;
  L2H_SCH) echo "--deep-head" ;;
  L2H_PCH) echo "--deep-head --client-heads" ;;
  *) echo "BADHEAD" ;; esac; }

sess_of(){ echo "wcc_${1}_w${2//./_}"; }   # tmux normalizes '.'->'_' in session names
sum_of(){  echo "results/csv/wcchead_${DS}_${1}_w${2}_summary.csv"; }
rows_of(){ [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }
is_done(){ [ "$(rows_of "$(sum_of "$1" "$2")")" -ge "$EXPECT_ROWS" ]; }

lane_count(){ # how many wcc_ sessions pinned to GPU $1
  local g=$1 s pid child cmd n=0
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^wcc_'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    child=$(pgrep -P "$pid" 2>/dev/null | head -1)
    cmd=$(ps -o args= -p "$child" 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && n=$((n+1))
  done
  echo "$n"
}

launch(){ # head lambda gpu
  local H=$1 L=$2 g=$3 sess hf
  sess=$(sess_of "$H" "$L"); hf=$(head_flags "$H")
  log "GPU $g <- $H lambda=$L  (tmux $sess)"
  tmux new -s "$sess" -d "cd $ML && \
    $PY train_split.py --sweep --dataset $DS --epochs $EPOCHS --num-clients $NC \
      --modes $MODES --block-size $BLOCK --compression-ratio-list $CRS --alpha-list $ALPHA \
      $hf --wcc-lambda $L --no-viz --gpu $g \
      --summary-csv $(sum_of "$H" "$L") \
      --epoch-csv results/csv/wcchead_${DS}_${H}_w${L}_epochs.csv 2>&1 | tee logs/${sess}.log"
}

JOBS=()
for H in "${HEADS[@]}"; do for L in "${LAMBDAS[@]}"; do JOBS+=("$H:$L"); done; done
log "WCC heads grid: ${#JOBS[@]} jobs (x${EXPECT_ROWS} rows), lanes=[${GPUS[*]}]x${PER_GPU}"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r H L <<< "$j"
    tmux has-session -t "$(sess_of "$H" "$L")" 2>/dev/null && continue
    is_done "$H" "$L" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all jobs done/running; exiting."; break; }
  free_gpu=""
  for g in "${GPUS[@]}"; do [ "$(lane_count "$g")" -lt "$PER_GPU" ] && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r H L <<< "${pending[0]}"
    launch "$H" "$L" "$free_gpu"
    sleep 15
  else
    sleep 20
  fi
done
log "WCC heads grid dispatcher exiting."
