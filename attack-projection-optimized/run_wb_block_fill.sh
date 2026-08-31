#!/usr/bin/env bash
# Fill the MISSING privileged wb_block cells so blocks 1/4/8 match block 2's full
# WCC-lambda grid {0,.001,.003,.01,.03,.1,.3,1,3,10} at CR{8,32}. Resume-guarded:
# a (block,lambda) cell with a non-empty summary.csv is skipped, so this only runs gaps.
# Mirrors run_wb_block_sweep.sh (--split-epochs 100 --retrain --experiments 3 4 --deep-head).
set -uo pipefail
PA="$(cd "$(dirname "$0")" && pwd)"
cd "$PA"
PY="${PYTHON:-python}"
GPUS=(${GPUS:-2 3})
mkdir -p "$PA/logs"
log(){ echo "[wbfill $(date '+%F %T')] $*"; }

# exactly the missing cells (block:lambda)
JOBS=(
  "1:0.3" "1:10.0"
  "8:0.3" "8:10.0"
  "4:0" "4:0.3" "4:1.0" "4:3.0" "4:10.0"
)

sess_of(){ local l="${2//./_}"; echo "wb_b${1}_l${l}"; }
out_of(){  echo "results/wb_block_b${1}_l${2}"; }
is_done(){ [ -s "$(out_of "$1" "$2")/summary.csv" ]; }

lane_free(){
  local g=$1 s pid cmd
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^wb_b'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && return 1
  done
  return 0
}

launch(){
  local B=$1 L=$2 g=$3 sess out
  sess=$(sess_of "$B" "$L"); out=$(out_of "$B" "$L")
  log "GPU $g <- block=$B lambda=$L  (tmux $sess, out $out)"
  tmux new -s "$sess" -d "cd $PA && $PY -u privileged_attack.py \
    --mode block_random --block-size $B --cr-list 8 32 \
    --experiments 3 4 --wcc-lambda $L --deep-head \
    --split-epochs 100 --retrain --gpu $g \
    --output-dir $out 2>&1 | tee logs/wb_b${B}_l${L}.log"
}

log "fill sweep: ${#JOBS[@]} cells, lanes=[${GPUS[*]}] (resume-guarded)"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r B L <<< "$j"
    tmux has-session -t "$(sess_of "$B" "$L")" 2>/dev/null && continue
    is_done "$B" "$L" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all fill cells done/running; exiting."; break; }
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
log "fill dispatcher exiting."
