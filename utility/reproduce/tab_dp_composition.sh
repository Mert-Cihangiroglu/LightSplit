#!/usr/bin/env bash
# Table tab:dp_utility / tab:dp_block (merged DP-composition utility table).
# Every arm adds per-sample L2 clipping (clip 10) + Gaussian noise (sigma 0.02):
#   Raw+DP           (16 runs x 2 distributions)
#   LightSplit-F/L+DP global (fr / frll, CR {8,16,32})
#   block-local b {1,2,4} + DP (transpose lift-back)
# One --sweep process per (dataset, head config, arm); 2000 epochs / patience 100
# (Raw+DP on CIFAR-10: 100 epochs, matching its no-DP twin). Several GPU-weeks.
# The no-DP references are the full utility sweep (run tab_full_sweep.sh) and,
# for the block rows, the block utility grid (run tab_block_util_FL.sh).
# SMOKE=1 runs two cells at 2 epochs.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=dp
JOBS="${JOBS:-1}"
DATASETS="${DATASETS:-cifar10 cifar100 fmnist gtsrb}"
HEADS="${HEADS:-L1H_SCH L1H_PCH L2H_SCH L2H_PCH}"
ARMS="${ARMS:-raw fr frll b1 b2 b4}"
CRS="${CRS:-8 16 32}"
SIG=0.02; CLIP=10.0; EXPECT_DP=6
if [ "${SMOKE:-0}" = 1 ]; then DATASETS=cifar10; HEADS=L2H_SCH; ARMS="raw b2"; CRS=8; fi
mkdir -p "$OUT/$FIG"
rows() { [ -f "$1" ] && tail -n +2 "$1" | grep -c . || echo 0; }
runq() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done; ( "$@" ) & }

for ds in $DATASETS; do
  case "$ds" in cifar100) NA=0.5 ;; *) NA=0.1 ;; esac
  ALPHAS="10000000 $NA"; EXPECT=$EXPECT_DP
  [ "${SMOKE:-0}" = 1 ] && { ALPHAS="10000000"; EXPECT=1; }
  for H in $HEADS; do
    flags=(); [ "${H%%_*}" = L2H ] && flags+=(--deep-head); [ "${H##*_}" = PCH ] && flags+=(--client-heads)
    for arm in $ARMS; do
      case "$arm" in
        raw)  mode=raw; extra=(); tag="dpraw_sig${SIG}_${ds}_${H}"; exp=2 ;;
        fr)   mode=fixed_random; extra=(); tag="dpblk_sig${SIG}_${ds}_${H}_fr"; exp=$EXPECT ;;
        frll) mode=fixed_random_ll; extra=(); tag="dpblk_sig${SIG}_${ds}_${H}_frll"; exp=$EXPECT ;;
        b*)   mode=block_random; extra=(--block-size "${arm#b}"); tag="dpblk_sig${SIG}_${ds}_${H}_${arm}"; exp=$EXPECT ;;
      esac
      [ "$ds" = gtsrb ] && [ "$arm" = b4 ] && continue
      EP=2000; PAT=100
      [ "$arm" = raw ] && [ "$ds" = cifar10 ] && EP=100
      [ "${SMOKE:-0}" = 1 ] && { EP=2; PAT=0; exp=1; }
      sum="$OUT/$FIG/${tag}_summary.csv"
      if [ "$(rows "$sum")" -ge "$exp" ]; then echo "[skip] $tag"; continue; fi
      echo "[$(date +%H:%M:%S)] $tag"
      runq "$PY" "$CODE_DIR/train_split.py" --sweep --dataset "$ds" --data-root "$DATA_ROOT" \
        --modes "$mode" ${extra[@]+"${extra[@]}"} --compression-ratio-list $CRS \
        --alpha-list $ALPHAS --num-clients 10 --epochs "$EP" --patience "$PAT" \
        --clip-norm "$CLIP" --noise-multiplier "$SIG" ${flags[@]+"${flags[@]}"} \
        --gpu 0 --no-viz --summary-csv "$sum" --epoch-csv "$OUT/$FIG/${tag}_epochs.csv"
    done
  done
done
wait
"${AGG_PYTHON:-$PY}" "$REPRO_DIR/aggregate_block_dp.py" \
  --util2k "$OUT/blockutil" --dpraw "$OUT/$FIG" --dpblk "$OUT/$FIG" \
  --fullsweep "$OUT/tab_full_sweep" --out-dp "$OUT/tab_dp_composition.csv"
