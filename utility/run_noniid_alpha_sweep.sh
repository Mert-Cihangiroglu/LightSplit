#!/usr/bin/env bash
# ============================================================================
# Non-IID heterogeneity (Dirichlet alpha) sweep, CIFAR-10, all head configs.
# ----------------------------------------------------------------------------
# Deliverable: utility (global + per-client acc) vs Dirichlet alpha for
#     * raw           -- reference (no defense), CR ignored
#     * block_random  -- OUR method, block in {1,2,4,8} (the privacy/utility dial)
#                        block 1 = max utility ... block 8 = global dense R
#                        (== fixed_random / LightSplit-F, max privacy endpoint).
#
# alpha sweep (non-IID -> mild):  0.1 0.3 1 3 10   (high alphas 30..1000 dropped: near-IID, not needed)
#   alpha small  -> strong label skew (hard non-IID);  alpha large -> ~IID.
# Head configs: L1H_SCH, L1H_PCH, L2H_SCH, L2H_PCH (controlled by HEADS env var).
# Setting: 10 clients, resnet18, 2000 epochs, patience 100, CR {8,16,32}, WCC 0.
#
# train_split --sweep iterates modes x CR x alpha internally, so:
#     * raw          : 1 process per head, sweeps alpha         -> 9 rows each
#     * block_random : 1 process per (head,block), sweeps CR x alpha -> 27 rows each
# Resume-safe: train_split --resume skips done cells; dispatcher skips full
#   summaries / live tmux sessions.  Co-locates CAP jobs/GPU.
# Detach:  tmux new -s nas2 -d "GPUS='0 1 2 3' CAP=2 HEADS='L1H_SCH L1H_PCH L2H_PCH' SEEDS='0 1 2' ./run_noniid_alpha_sweep.sh"
# L2H_SCH note: the original single-seed run (nas_raw, nas_br_b*) writes to
#   CSVs without seed suffix; new seeded runs use _s{seed} in both session
#   names and CSV paths so they are fully independent.
# Smoke:   ALPHAS='0.1 1000' EPOCHS=2 CRS='8' BLOCKS='1 8' HEADS='L1H_SCH' SEEDS='0' ./run_noniid_alpha_sweep.sh
# ============================================================================
set -uo pipefail
ML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # derive from script location
cd "$ML" || { echo "FATAL: cannot cd to script dir $ML"; exit 1; }
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3})
CAP="${CAP:-2}"                                  # max concurrent jobs per GPU
BLOCKS=(${BLOCKS:-1 2 4 8})
CRS="${CRS:-8 16 32}"
ALPHAS="${ALPHAS:-0.1 0.3 1 3 10}"
EPOCHS="${EPOCHS:-2000}"
PATIENCE="${PATIENCE:-100}"
NC="${NC:-10}"
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH})
SEEDS=(${SEEDS:-0 1 2})
mkdir -p "$ML/logs" "$ML/results/csv"
log(){ echo "[nas $(date '+%F %T')] $*"; }

head_flags(){ case "$1" in
  L1H_SCH) echo "" ;;
  L1H_PCH) echo "--client-heads" ;;
  L2H_SCH) echo "--deep-head" ;;
  L2H_PCH) echo "--deep-head --client-heads" ;;
  *) echo "BADHEAD" ;; esac; }

n_alpha(){ set -- $ALPHAS; echo "$#"; }
n_cr(){ set -- $CRS; echo "$#"; }

# ---- job list: "fam:block:htag:seed" (block '-' for raw) ----
JOBS=()
for S in "${SEEDS[@]}"; do
  for H in "${HEADS[@]}"; do
    JOBS+=("raw:-:$H:$S")
    for B in "${BLOCKS[@]}"; do JOBS+=("br:$B:$H:$S"); done
  done
done

sess_of(){  # fam block htag seed
  if [ "$1" = raw ]; then echo "nas_raw_${3}_s${4}"; else echo "nas_br_b${2}_${3}_s${4}"; fi; }
sum_of(){   # fam block htag seed
  if [ "$1" = raw ]; then echo "results/csv/noniid_alpha_raw_cifar10_${3}_s${4}_summary.csv";
  else echo "results/csv/noniid_alpha_block_random_b${2}_cifar10_${3}_s${4}_summary.csv"; fi; }
epo_of(){   # fam block htag seed
  if [ "$1" = raw ]; then echo "results/csv/noniid_alpha_raw_cifar10_${3}_s${4}_epochs.csv";
  else echo "results/csv/noniid_alpha_block_random_b${2}_cifar10_${3}_s${4}_epochs.csv"; fi; }
expect_of(){ if [ "$1" = raw ]; then n_alpha; else echo "$(( $(n_alpha) * $(n_cr) ))"; fi; }
rows_of(){ [ -f "$1" ] && awk 'END{print NR-1}' "$1" 2>/dev/null || echo 0; }
is_done(){ local f; f="$(sum_of "$1" "$2" "$3" "$4")"; [ "$(rows_of "$f")" -ge "$(expect_of "$1")" ]; }

lane_count(){ # gpu -> number of running nas_* sessions pinned to it
  local g=$1 s pid cmd n=0
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^nas_'); do
    pid=$(tmux list-panes -t "$s" -F '#{pane_pid}' 2>/dev/null)
    cmd=$(pgrep -P "$pid" 2>/dev/null | head -1 | xargs -r -I{} ps -o args= -p {} 2>/dev/null)
    echo "$cmd" | grep -qE -- "--gpu $g( |\$)" && n=$((n+1))
  done
  echo "$n"; }

launch(){ # fam block htag seed gpu
  local fam=$1 B=$2 H=$3 S=$4 g=$5 sess cmd hf
  sess=$(sess_of "$fam" "$B" "$H" "$S")
  hf=$(head_flags "$H")
  if [ "$fam" = raw ]; then
    cmd="$PY -u train_split.py --sweep --dataset cifar10 --modes raw \
      --compression-ratio-list $CRS --alpha-list $ALPHAS --num-clients $NC \
      --epochs $EPOCHS --patience $PATIENCE --seed $S --gpu $g $hf --no-viz --resume \
      --summary-csv $(sum_of "$fam" "$B" "$H" "$S") --epoch-csv $(epo_of "$fam" "$B" "$H" "$S")"
    log "GPU $g <- raw seed=$S (${EPOCHS}ep, alpha sweep) ${H}  (tmux $sess)"
  else
    cmd="$PY -u train_split.py --sweep --dataset cifar10 --modes block_random --block-size $B \
      --compression-ratio-list $CRS --alpha-list $ALPHAS --num-clients $NC \
      --epochs $EPOCHS --patience $PATIENCE --seed $S --gpu $g $hf --no-viz --resume \
      --summary-csv $(sum_of "$fam" "$B" "$H" "$S") --epoch-csv $(epo_of "$fam" "$B" "$H" "$S")"
    log "GPU $g <- block_random b$B seed=$S (${EPOCHS}ep, alpha sweep) ${H}  (tmux $sess)"
  fi
  tmux new -s "$sess" -d "cd $ML && $cmd 2>&1 | tee logs/${sess}.log"; }

n_jobs=${#JOBS[@]}
log "non-iid alpha sweep cifar10: $n_jobs jobs across heads=[${HEADS[*]}] seeds=[${SEEDS[*]}], lanes=[${GPUS[*]}] cap=$CAP/gpu"
log "  alphas=[$ALPHAS] blocks=[${BLOCKS[*]}] CR=[$CRS] epochs=$EPOCHS patience=$PATIENCE"
while true; do
  pending=()
  for j in "${JOBS[@]}"; do
    IFS=: read -r fam B H S <<< "$j"
    tmux has-session -t "$(sess_of "$fam" "$B" "$H" "$S")" 2>/dev/null && continue
    is_done "$fam" "$B" "$H" "$S" && continue
    pending+=("$j")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all jobs done/running; dispatcher exiting."; break; }

  free_gpu=""
  for g in "${GPUS[@]}"; do [ "$(lane_count "$g")" -lt "$CAP" ] && { free_gpu=$g; break; }; done
  if [ -n "$free_gpu" ]; then
    IFS=: read -r fam B H S <<< "${pending[0]}"
    launch "$fam" "$B" "$H" "$S" "$free_gpu"
    sleep 15
  else
    sleep 30
  fi
done
log "non-iid alpha sweep cifar10 dispatcher exiting (pending=0)."
