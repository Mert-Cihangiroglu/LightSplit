#!/usr/bin/env bash
# ============================================================================
# block_random {b1,b2,b4} + fixed_random (= b8, global endpoint) WITH DP
# (per-sample L2 clip=10 + Gaussian noise, sigma=0.02) across:
#   4 datasets x 4 head configs x CR{8,16,32} x {IID, non-IID}
# non-IID alpha = 0.5 for cifar100, 0.1 otherwise (table-canonical).
# 2000ep / patience 100 / 10 clients — matches the util2k grid so the DP cells
# are directly comparable to the no-DP block_random grid.
#
# clip=10 verified appropriate: median ||payload|| = 58 (raw) / 19.6 (fixed_random)
# / 19–21 (block_random b1/b2/b4) in the utility harness, so clipping bites on 100% of
# samples for every mode (the FORA "tiny z" caveat does not apply here).
#
# gtsrb skips b4 (14x14 spatial not divisible by 4 -> BlockRandomBottleneck raises)
# and runs on LOCAL NVMe (sshfs per-epoch image reads are I/O-bound).
# --sweep does CR x alpha in one process -> 6 rows/cell. Resume-safe.
# Detach: tmux new -s dpblk -d "bash run_dp_block_table.sh"
# ============================================================================
set -uo pipefail
MNT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3}); NG=${#GPUS[@]}
MAXCONC="${MAXCONC:-12}"            # GPUs are free -> 3/GPU like the util2k grid
SIG="${SIG:-0.02}"; CLIP="${CLIP:-10.0}"; EPOCHS="${EPOCHS:-2000}"; PAT="${PAT:-100}"; NC="${NC:-10}"
CRS="${CRS:-8 16 32}"
DATASETS=(${DATASETS:-cifar10 cifar100 fmnist gtsrb})
HEADS=(${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH})
GTSRB_LOCAL="${GTSRB_LOCAL:-$ML/data/gtsrb}"
log(){ echo "[dpblk $(date '+%F %T')] $*"; }

niid_alpha(){ case "$1" in cifar100) echo 0.5;; *) echo 0.1;; esac; }
head_flags(){ case "$1" in
  L1H_SCH) echo "";; L1H_PCH) echo "--client-heads";;
  L2H_SCH) echo "--deep-head";; L2H_PCH) echo "--deep-head --client-heads";; esac; }
# gtsrb: 14x14 not divisible by 4 -> no b4.  MODES env overrides (e.g. MODES=frll for LS-L only).
modes_of(){ [ -n "${MODES:-}" ] && { echo "$MODES"; return; }
  case "$1" in gtsrb) echo "b1 b2 fr";; *) echo "b1 b2 b4 fr";; esac; }
mode_args(){ case "$1" in
  fr)   echo "--modes fixed_random";;        # LightSplit-F (transpose lift-back)
  frll) echo "--modes fixed_random_ll";;     # LightSplit-L (server MLP lift-back)
  b*)   echo "--modes block_random --block-size ${1#b}";; esac; }
wd_of(){  [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL" || echo "$MNT"; }
dr_of(){  [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL/data" || echo "./data"; }
out_of(){ [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL/results/csv" || echo "results/csv"; }
sumbase(){ echo "dpblk_sig${SIG}_${1}_${2}_${3}_summary.csv"; }   # ds head modetag

n_expect(){ set -- $CRS; echo $(( $# * 2 )); }   # CR x {IID, non-IID}
is_done(){ local f="$MNT/results/csv/$(sumbase "$1" "$2" "$3")"
  [ -f "$f" ] && [ "$(($(wc -l <"$f")-1))" -ge "$(n_expect)" ]; }
sess_of(){ echo "dpblk_${1}_${2}_${3}"; }

launch(){ # ds head modetag gpu
  local DS=$1 H=$2 M=$3 g=$4 wd dr out sb sess cmd
  wd=$(wd_of "$DS"); dr=$(dr_of "$DS"); out=$(out_of "$DS"); sb=$(sumbase "$DS" "$H" "$M"); sess=$(sess_of "$DS" "$H" "$M")
  mkdir -p "$wd/logs" "$wd/$out" 2>/dev/null
  cmd="cd $wd && $PY -u $MNT/train_split.py --sweep --dataset $DS $(mode_args "$M") \
    --compression-ratio-list $CRS --alpha-list 10000000 $(niid_alpha "$DS") --num-clients $NC \
    --epochs $EPOCHS --patience $PAT --clip-norm $CLIP --noise-multiplier $SIG \
    --data-root $dr $(head_flags "$H") --no-viz --resume --gpu $g \
    --summary-csv $out/$sb --epoch-csv $out/dpblk_sig${SIG}_${DS}_${H}_${M}_epochs.csv"
  [ "$DS" = gtsrb ] && cmd="$cmd; cp -f $out/$sb $MNT/results/csv/$sb"
  log "GPU $g <- $DS $H $M (niid a=$(niid_alpha "$DS"))  tmux $sess"
  tmux new -s "$sess" -d "$cmd 2>&1 | tee $wd/logs/${sess}.log"
}

JOBS=(); for DS in "${DATASETS[@]}"; do for H in "${HEADS[@]}"; do for M in $(modes_of "$DS"); do JOBS+=("$DS:$H:$M"); done; done; done
log "dp block-random table: ${#JOBS[@]} cells x $(n_expect) configs, sig=$SIG clip=$CLIP ep=$EPOCHS maxconc=$MAXCONC"
rr=0
while true; do
  running=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -c '^dpblk_[a-z]')
  pending=()
  for j in "${JOBS[@]}"; do IFS=: read -r DS H M <<< "$j"
    tmux has-session -t "$(sess_of "$DS" "$H" "$M")" 2>/dev/null && continue
    is_done "$DS" "$H" "$M" && continue
    pending+=("$j"); done
  [ ${#pending[@]} -eq 0 ] && { log "all dp-block cells done/running; dispatcher exiting."; break; }
  if [ "$running" -lt "$MAXCONC" ]; then
    IFS=: read -r DS H M <<< "${pending[0]}"
    launch "$DS" "$H" "$M" "${GPUS[$((rr % NG))]}"; rr=$((rr+1)); sleep 8
  else sleep 30; fi
done
log "dp block-random table dispatcher exiting."
