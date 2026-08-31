#!/usr/bin/env bash
# White-box known-R block_random dial sweep (privacy quantification, CIFAR-10).
# Mirrors run_wb_ceiling.sh but sweeps the block dial x WCC lambda:
#   block in {1,2,4,8}  (8 == fixed_random sanity vs existing ceiling; 1 == most-local/least-private)
#   lambda in {0, 0.1}  (0.1 = decisive privacy cell; 0 = baseline)
#   --cr-list 8 32  --experiments 3 4  --deep-head  --split-epochs 100  --retrain
# Resume-guarded: a (block,lambda) cell whose output summary.csv is non-empty is skipped.
# Co-locates on shared GPUs (grid uses ~3/48 GB/card). Override lanes: GPUS="0 1 2 3" bash run_wb_block_sweep.sh
# Detach: run this whole script inside tmux 'wbq'; each cell gets its own tmux wb_b{B}_l{L}.
set -uo pipefail
PA="$(cd "$(dirname "$0")" && pwd)"
cd "$PA"
PY="${PYTHON:-python}"
GPUS=(${GPUS:-2 3})                             # GPU lanes to co-locate on
mkdir -p "$PA/logs"
log(){ echo "[wbq $(date '+%F %T')] $*"; }

# priority: b8 sanity (==fixed_random) + b2 operating point first, lambda=0.1 (decisive) before 0
JOBS=(
  # fixed_random (== b8 global scramble) full WCC ladder — completes the CONTROLLED
  # SDAR(no-R) vs privileged(known-R) comparison on the SAME bottleneck. Decisive cell
  # 8:0.01 = where SDAR fixed_random craters to the 0.374 prototype floor (92% acc kept);
  # checks whether privileged known-R instead stays flat at ~0.5. CR8+CR32 per cell.
  "8:0.01" "8:0.001" "8:0.003" "8:0.03" "8:1.0" "8:3.0"
  # CORRECTED small-lambda WCC ladder (graceful regime; knee ~0.01). Target = the 0.37
  # class-prototype SSIM floor, NOT random. b2 first.
  "2:0.01" "2:0.003" "2:0.03" "2:0.001"
  "1:0.01" "1:0.003" "1:0.03" "1:0.001"
  "4:0.01" "4:0.003" "4:0.03" "4:0.001"
  # reference (resume-guard skips finished): 0.1 anchor + 0 baseline (full dial incl b8)
  "8:0.1" "2:0.1" "1:0.1" "4:0.1"  "8:0" "2:0" "1:0" "4:0"
)

sess_of(){ local l="${2//./_}"; echo "wb_b${1}_l${l}"; }   # tmux rewrites '.'->'_' in session names; match that
out_of(){  echo "results/wb_block_b${1}_l${2}"; }
is_done(){ [ -s "$(out_of "$1" "$2")/summary.csv" ]; }     # non-empty summary = complete cell

lane_free(){ # gpu -> return 0 (free) if no running wb_* session is pinned to it
  local g=$1 s pid cmd
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^wb_b'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && return 1
  done
  return 0
}

launch(){ # B L gpu
  local B=$1 L=$2 g=$3 sess out
  sess=$(sess_of "$B" "$L"); out=$(out_of "$B" "$L")
  log "GPU $g <- block=$B lambda=$L  (tmux $sess, out $out)"
  tmux new -s "$sess" -d "cd $PA && $PY -u privileged_attack.py \
    --mode block_random --block-size $B --cr-list 8 32 \
    --experiments 3 4 --wcc-lambda $L --deep-head \
    --split-epochs 100 --retrain --gpu $g \
    --output-dir $out 2>&1 | tee logs/wb_b${B}_l${L}.log"
}

log "block sweep: ${#JOBS[@]} cells, lanes=[${GPUS[*]}]  (resume-guarded)"
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
    sleep 20      # let the cell claim its tmux session before re-scanning
  else
    sleep 30
  fi
done
log "sweep dispatcher exiting."
