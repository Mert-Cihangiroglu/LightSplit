#!/usr/bin/env bash
# Reseed: rerun the FULL privileged known-R wb_block grid at additional seeds so
# every published cell gets a mean+/-std (n=3 with the existing seed-42 grid).
#
# Faithful replica of the two original passes, per seed:
#   arm main : --cr-list 8 32  -> results/wb_block_b{B}_l{L}_s{SEED}
#   arm cr16 : --cr-list 16    -> results/wb_block_b{B}_l{L}_cr16_s{SEED}
# with --mode block_random --experiments 3 4 --deep-head --split-epochs 100 --retrain,
# exactly as run_wb_block_sweep.sh / run_wb_block_cr16.sh. New --seed => new R, new
# 90/10 data split, new decoder init (torch-seeded), i.e. an independent trajectory.
#
# The seed-42 dirs (no _s suffix) are the original grid and are NEVER touched.
# Resume-guarded on ROW COMPLETENESS, not mere file existence: a cell counts as done
# only when summary.csv holds the expected number of exp3 rows (main=2 [CR8+CR32],
# cr16=1). This closes the STATUS.md footgun where a non-empty-but-partial summary was
# skipped and a CR row was silently lost.
#
# Co-locates on all 4 idle GPUs; each cell runs in its own tmux (wbr_* / wbr16_*), so a
# dead dispatcher does not kill in-flight cells. Detach the dispatcher itself in tmux 'wbr'.
set -uo pipefail
PA="$(cd "$(dirname "$0")" && pwd)"
cd "$PA"
PY="${PYTHON:-python}"
GPUS=(${GPUS:-0 1 2 3})
SEEDS=(${SEEDS:-43 44})
LAMS=(${LAMS:-0 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 10.0})
BLOCKS=(${BLOCKS:-1 2 4 8})
mkdir -p "$PA/logs"
log(){ echo "[wbr $(date '+%F %T')] $*"; }

# job = "SEED:B:L:ARM"  (ARM in {main,cr16})
JOBS=()
for S in "${SEEDS[@]}"; do
  for B in "${BLOCKS[@]}"; do
    for L in "${LAMS[@]}"; do
      JOBS+=("$S:$B:$L:main" "$S:$B:$L:cr16")
    done
  done
done

lsan(){ echo "${1//./_}"; }                       # tmux rewrites '.' -> '_' in session names
sess_of(){ # S B L ARM
  local s=$1 b=$2 l; l=$(lsan "$3")
  [ "$4" = cr16 ] && echo "wbr16_s${s}_b${b}_l${l}" || echo "wbr_s${s}_b${b}_l${l}"
}
out_of(){ # S B L ARM
  [ "$4" = cr16 ] && echo "results/wb_block_b${2}_l${3}_cr16_s${1}" \
                  || echo "results/wb_block_b${2}_l${3}_s${1}"
}
need_of(){ [ "$1" = cr16 ] && echo 1 || echo 2; }  # expected exp3 rows
is_done(){ # dir need
  [ -s "$1/summary.csv" ] || return 1
  local n; n=$(grep -c '^exp3_known' "$1/summary.csv" 2>/dev/null)
  [ "${n:-0}" -ge "$2" ]
}

lane_free(){ # gpu -> 0 if no running wbr* session pinned to it
  local g=$1 s pid cmd
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^wbr(16)?_s'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && return 1
  done
  return 0
}

launch(){ # S B L ARM gpu
  local S=$1 B=$2 L=$3 ARM=$4 g=$5 sess out crlist logf
  sess=$(sess_of "$S" "$B" "$L" "$ARM"); out=$(out_of "$S" "$B" "$L" "$ARM")
  [ "$ARM" = cr16 ] && { crlist="16"; logf="logs/wbr16_s${S}_b${B}_l${L}.log"; } \
                    || { crlist="8 32"; logf="logs/wbr_s${S}_b${B}_l${L}.log"; }
  log "GPU $g <- seed=$S block=$B lambda=$L cr=[$crlist]  (tmux $sess, out $out)"
  tmux new -s "$sess" -d "cd $PA && $PY -u privileged_attack.py \
    --mode block_random --block-size $B --cr-list $crlist \
    --experiments 3 4 --wcc-lambda $L --deep-head --seed $S \
    --split-epochs 100 --retrain --gpu $g \
    --output-dir $out 2>&1 | tee $logf"
}

log "reseed grid: ${#JOBS[@]} cell-runs (seeds=[${SEEDS[*]}] blocks=[${BLOCKS[*]}] lam=${#LAMS[@]} x2 arms), lanes=[${GPUS[*]}]"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r S B L ARM <<< "$j"
    tmux has-session -t "$(sess_of "$S" "$B" "$L" "$ARM")" 2>/dev/null && continue
    is_done "$(out_of "$S" "$B" "$L" "$ARM")" "$(need_of "$ARM")" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all reseed cells done/running; dispatcher exiting."; break; }

  free_gpu=""
  for g in "${GPUS[@]}"; do lane_free "$g" && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r S B L ARM <<< "${pending[0]}"
    launch "$S" "$B" "$L" "$ARM" "$free_gpu"
    sleep 20      # let the cell claim its tmux session before re-scanning
  else
    sleep 30
  fi
done
log "reseed dispatcher exiting (${#JOBS[@]} jobs, ${#SEEDS[@]} seeds)."
