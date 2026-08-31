#!/usr/bin/env bash
# ============================================================================
# raw + DP (Gaussian, sigma=0.02, clip=10) across ALL utility datasets x ALL
# head configs x {IID, non-IID} -- the "DP+RAW" rows for the giant utility table.
#   * mode = raw ONLY (no LightSplit / fixed_random).
#   * heads: L1H_SCH L1H_PCH L2H_SCH L2H_PCH.
#   * dists: IID (alpha=1e7) + non-IID (alpha=0.5 for cifar100, 0.1 otherwise).
#   * 2000ep patience 100, 10 clients  (== the converged utility table's budget).
# --sweep does IID+non-IID in one process (2 rows/cell). 4 ds x 4 heads = 16 cells.
# gtsrb runs on LOCAL NVMe (sshfs per-epoch image reads are I/O-bound); its summary
# is copied back to the mount. Resume-safe. Detach: tmux new -s dpraw_disp -d "bash run_dp_raw_table.sh"
# ============================================================================
set -uo pipefail
MNT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # this folder
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3}); NG=${#GPUS[@]}
MAXCONC="${MAXCONC:-4}"                                  # total concurrent jobs (co-located w/ noniid)
SIG="${SIG:-0.02}"; CLIP="${CLIP:-10.0}"; EPOCHS="${EPOCHS:-2000}"; PAT="${PAT:-100}"; NC="${NC:-10}"
DATASETS=(${DATASETS:-cifar10 cifar100 fmnist gtsrb})
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH})
GTSRB_LOCAL="${GTSRB_LOCAL:-$ML/data/gtsrb}"
log(){ echo "[dpraw $(date '+%F %T')] $*"; }

niid_alpha(){ case "$1" in cifar100) echo 0.5;; *) echo 0.1;; esac; }
head_flags(){ case "$1" in
  L1H_SCH) echo "";; L1H_PCH) echo "--client-heads";;
  L2H_SCH) echo "--deep-head";; L2H_PCH) echo "--deep-head --client-heads";; esac; }
# per-dataset working dir + data root + output dir (gtsrb -> local NVMe)
wd_of(){ [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL" || echo "$MNT"; }
dr_of(){ [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL/data" || echo "./data"; }
out_of(){ [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL/results/csv" || echo "results/csv"; }
sumbase(){ echo "dpraw_sig${SIG}_${1}_${2}_summary.csv"; }

is_done(){ # dataset head -> summary has both rows
  local f="$MNT/results/csv/$(sumbase "$1" "$2")"
  [ -f "$f" ] && [ "$(($(wc -l < "$f")-1))" -ge 2 ]; }
sess_of(){ echo "dpraw_${1}_${2}"; }

launch(){ # dataset head gpu
  local DS=$1 H=$2 g=$3 wd dr out hf sb sess cmd
  wd=$(wd_of "$DS"); dr=$(dr_of "$DS"); out=$(out_of "$DS"); hf=$(head_flags "$H")
  sb=$(sumbase "$DS" "$H"); sess=$(sess_of "$DS" "$H")
  mkdir -p "$wd/logs" "$wd/$out" 2>/dev/null
  cmd="cd $wd && $PY -u $MNT/train_split.py --sweep --dataset $DS --modes raw \
    --compression-ratio-list 8 --alpha-list 10000000 $(niid_alpha "$DS") --num-clients $NC \
    --epochs $EPOCHS --patience $PAT --clip-norm $CLIP --noise-multiplier $SIG \
    --data-root $dr $hf --no-viz --resume --gpu $g \
    --summary-csv $out/$sb --epoch-csv $out/dpraw_sig${SIG}_${DS}_${H}_epochs.csv"
  [ "$DS" = gtsrb ] && cmd="$cmd; cp -f $out/$sb $MNT/results/csv/$sb"   # copy gtsrb summary back to mount
  log "GPU $g <- $DS $H  (niid a=$(niid_alpha "$DS"))  tmux $sess"
  tmux new -s "$sess" -d "$cmd 2>&1 | tee $wd/logs/${sess}.log"
}

JOBS=(); for DS in "${DATASETS[@]}"; do for H in "${HEADS[@]}"; do JOBS+=("$DS:$H"); done; done
log "dp-raw utility-table sweep: ${#JOBS[@]} cells, maxconc=$MAXCONC, gpus=[${GPUS[*]}], sig=$SIG ep=$EPOCHS"
rr=0
while true; do
  running=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -c '^dpraw_[a-z]')
  pending=()
  for j in "${JOBS[@]}"; do IFS=: read -r DS H <<< "$j"
    tmux has-session -t "$(sess_of "$DS" "$H")" 2>/dev/null && continue
    is_done "$DS" "$H" && continue
    pending+=("$j"); done
  [ ${#pending[@]} -eq 0 ] && { log "all dp-raw cells done/running; dispatcher exiting."; break; }
  if [ "$running" -lt "$MAXCONC" ]; then
    IFS=: read -r DS H <<< "${pending[0]}"
    launch "$DS" "$H" "${GPUS[$((rr % NG))]}"; rr=$((rr+1)); sleep 10
  else sleep 30; fi
done
log "dp-raw utility-table sweep dispatcher exiting."
