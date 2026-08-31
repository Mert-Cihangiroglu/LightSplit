#!/usr/bin/env bash
# Add the missing CR16 to the privileged wb_block sweep (it was CR8/CR32 only).
# block {1,2,4,8} x full WCC-lambda grid at CR16 -> 40 cells. Written to SEPARATE
# dirs (wb_block_b{B}_l{L}_cr16) so the existing CR8/CR32 summaries are NOT clobbered
# (summary.csv is rewritten mode='w' per run). The plot scripts derive CR from k, so
# k=256 rows register as CR16 once the glob includes these dirs. Resume-guarded.
set -uo pipefail
PA="$(cd "$(dirname "$0")" && pwd)"
cd "$PA"
PY="${PYTHON:-python}"
GPUS=(${GPUS:-0 1})
LAMS=(${LAMS:-0 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 10.0})
BLOCKS=(${BLOCKS:-1 2 4 8})
mkdir -p "$PA/logs"
log(){ echo "[wb16 $(date '+%F %T')] $*"; }

JOBS=()
for B in "${BLOCKS[@]}"; do for L in "${LAMS[@]}"; do JOBS+=("$B:$L"); done; done

sess_of(){ local l="${2//./_}"; echo "wb16_b${1}_l${l}"; }
out_of(){  echo "results/wb_block_b${1}_l${2}_cr16"; }
is_done(){ [ -s "$(out_of "$1" "$2")/summary.csv" ]; }

lane_free(){
  local g=$1 s pid cmd
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^wb16_b'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && return 1
  done
  return 0
}

launch(){
  local B=$1 L=$2 g=$3 sess out
  sess=$(sess_of "$B" "$L"); out=$(out_of "$B" "$L")
  log "GPU $g <- block=$B lambda=$L CR16  (tmux $sess, out $out)"
  tmux new -s "$sess" -d "cd $PA && $PY -u privileged_attack.py \
    --mode block_random --block-size $B --cr-list 16 \
    --experiments 3 4 --wcc-lambda $L --deep-head \
    --split-epochs 100 --retrain --gpu $g \
    --output-dir $out 2>&1 | tee logs/wb16_b${B}_l${L}.log"
}

log "CR16 sweep: ${#JOBS[@]} cells, lanes=[${GPUS[*]}] (resume-guarded)"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r B L <<< "$j"
    tmux has-session -t "$(sess_of "$B" "$L")" 2>/dev/null && continue
    is_done "$B" "$L" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all CR16 cells done/running; exiting."; break; }
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
log "CR16 dispatcher exiting."
