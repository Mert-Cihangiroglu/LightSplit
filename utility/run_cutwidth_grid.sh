#!/usr/bin/env bash
# ============================================================================
# Reduced cut-layer-width baseline
#
# The client head emits cut_channels x H x W instead of the native C x H x W, so the
# smashed tensor is d = cut_channels*H*W.  Then compression is applied ON TOP
# (fixed_random / block_random b2) at CR{8,16,32}.  Cut width is a SEPARATE axis from
# CR; the `raw` (CR=1) row at each width isolates the effect of narrowing by itself.
#
# This is NOT learned_1x1: there the head keeps its native width and a 1x1 conv squeezes
# it for transmission.  Here the head's own weights shrink and there is no squeeze conv.
# (Corollary, verified: learned_1x1 CANNOT compress at narrow widths -- it only squeezes
# the channel axis, so k_channels = C_cut//CR floors to 1.  fixed_random has no such limit.)
#
# Grid: 4 datasets x 6 cut widths x {raw + fixed_random + block_random b2} x CR{8,16,32}
#       x {IID, non-IID}.  Head = L2H_SCH (deployed).  non-IID alpha = 0.5 cifar100, 0.1 else.
#
# Budget: 100 ep, full cosine (T_max=epochs) -- the X5 matched-byte convention, a CONVERGED
# budget (topk_sparse@100ep 85.66% > block_random@2000ep 85.48% on cifar10/L2H_SCH/IID/CR8).
# The native-width anchor already exists at 2000ep in util2k_* for cross-checking.
#
# SKIP GUARD: drop any (width, CR) with k = d // CR < 8 -- payload too small to mean anything.
# Dropped cells are logged, never silently omitted.
#
# CAVEATS recorded in the CSV (achieved payload_dim / compression_ratio), not assumed:
#   - block_random payload floors: payload = n_blocks * max(1, floor(k/n_blocks)), so at
#     narrow widths the ACHIEVED CR is lower than nominal.
#   - k is relative to the NARROWED d, so "CR8" means 8x of that width, not of 4096.
#     The honest cross-width axis is transmitted floats.
#
# gtsrb runs on LOCAL NVMe (sshfs per-epoch image reads are I/O-bound), summaries copied back.
# One process per (dataset, cut_channels); each --sweep emits all its rows into its OWN CSV,
# so cut widths never collide inside a file and --resume needs no new key.
#
# Detach:  tmux new -s cutwq -d "MAXCONC=2 bash run_cutwidth_grid.sh"
# NOTE: dpblkq/dpllq (X4) hold 12 lanes -- run at MAXCONC=2 until they drain.
# ============================================================================
set -uo pipefail
MNT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
GPUS=(${GPUS:-0 1 2 3}); NG=${#GPUS[@]}
MAXCONC="${MAXCONC:-2}"              # X4 owns the box; be a good neighbour
EPOCHS="${EPOCHS:-100}"; NC="${NC:-10}"
DATASETS=(${DATASETS:-cifar10 cifar100 fmnist gtsrb})
GTSRB_LOCAL="${GTSRB_LOCAL:-$ML/data/gtsrb}"
log(){ echo "[cutw $(date '+%F %T')] $*"; }

# ── per-dataset geometry ────────────────────────────────────────────────────
hw_of(){ case "$1" in cifar10|cifar100) echo 64;; fmnist) echo 144;; gtsrb) echo 196;; esac; }
# channel ladder ~= native / {1,2,4,8,16,32}
ladder_of(){ case "$1" in
  cifar10|cifar100) echo "64 32 16 8 4 2";;   # d = 4096 2048 1024 512 256 128
  fmnist)           echo "20 10 5 3 2 1";;    # d = 2880 1440 720 432 288 144
  gtsrb)            echo "29 15 8 4 2 1";;    # d = 5684 2940 1568 784 392 196
esac; }
niid_alpha(){ case "$1" in cifar100) echo 0.5;; *) echo 0.1;; esac; }

# CRs that survive the k>=8 skip guard at this width
crs_for(){ local d=$1 out=""
  for cr in 8 16 32; do [ $((d / cr)) -ge 8 ] && out="$out $cr"; done
  echo "${out# }"; }

wd_of(){  [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL" || echo "$MNT"; }
dr_of(){  [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL/data" || echo "./data"; }
out_of(){ [ "$1" = gtsrb ] && echo "$GTSRB_LOCAL/results/csv" || echo "results/csv"; }
sumbase(){ echo "cutw_${1}_L2H_SCH_c${2}_summary.csv"; }     # ds cut_channels
sess_of(){ echo "cutw_${1}_c${2}"; }

# rows = alphas{IID,nonIID} x (1 raw + 2 compressed modes x |CRS|)
n_expect(){ local ncr; ncr=$(wc -w <<< "$1"); echo $(( 2 * (1 + 2 * ncr) )); }
is_done(){ local f="$MNT/results/csv/$(sumbase "$1" "$2")"
  [ -f "$f" ] && [ "$(($(wc -l <"$f")-1))" -ge "$3" ]; }

launch(){ # ds cut_channels d crs gpu
  local DS=$1 C=$2 D=$3 CRS=$4 g=$5 wd dr out sb sess cmd
  wd=$(wd_of "$DS"); dr=$(dr_of "$DS"); out=$(out_of "$DS")
  sb=$(sumbase "$DS" "$C"); sess=$(sess_of "$DS" "$C")
  mkdir -p "$wd/logs" "$wd/$out" 2>/dev/null
  cmd="cd $wd && $PY -u $MNT/train_split.py --sweep --dataset $DS \
    --modes raw fixed_random block_random --block-size 2 \
    --cut-channels $C --compression-ratio-list $CRS \
    --alpha-list 10000000 $(niid_alpha "$DS") --num-clients $NC \
    --epochs $EPOCHS --data-root $dr --deep-head --no-viz --resume --gpu $g \
    --summary-csv $out/$sb --epoch-csv $out/cutw_${DS}_L2H_SCH_c${C}_epochs.csv"
  [ "$DS" = gtsrb ] && cmd="$cmd; cp -f $out/$sb $MNT/results/csv/$sb"
  log "GPU $g <- $DS c=$C (d=$D) CR={$CRS}  tmux $sess"
  tmux new -s "$sess" -d "$cmd 2>&1 | tee $wd/logs/${sess}.log"
}

# ── build job list, logging every skipped (width, CR) ───────────────────────
JOBS=()
for DS in "${DATASETS[@]}"; do
  HW=$(hw_of "$DS")
  for C in $(ladder_of "$DS"); do
    D=$(( C * HW )); CRS=$(crs_for "$D")
    for cr in 8 16 32; do
      grep -qw "$cr" <<< "$CRS" || log "SKIP $DS c=$C (d=$D) CR$cr -> k=$((D/cr)) < 8"
    done
    JOBS+=("$DS:$C:$D:$CRS")
  done
done
log "cut-width grid: ${#JOBS[@]} cells, ep=$EPOCHS head=L2H_SCH maxconc=$MAXCONC"

rr=0
while true; do
  # count workers only (session names start 'cutw_'); dispatcher is 'cutwq' -> no self-count
  running=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -c '^cutw_[a-z]')
  pending=()
  for j in "${JOBS[@]}"; do IFS=: read -r DS C D CRS <<< "$j"
    tmux has-session -t "$(sess_of "$DS" "$C")" 2>/dev/null && continue
    is_done "$DS" "$C" "$(n_expect "$CRS")" && continue
    pending+=("$j"); done
  [ ${#pending[@]} -eq 0 ] && { log "all cut-width cells done/running; dispatcher exiting."; break; }
  if [ "$running" -lt "$MAXCONC" ]; then
    IFS=: read -r DS C D CRS <<< "${pending[0]}"
    launch "$DS" "$C" "$D" "$CRS" "${GPUS[$((rr % NG))]}"; rr=$((rr+1)); sleep 8
  else sleep 30; fi
done
log "cut-width dispatcher exiting."
