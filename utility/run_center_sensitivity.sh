#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Center loss (Wen et al. 2016) vs WCC -- UTILITY sweep (multi-client harness).
#
# Both losses are the SAME class-compaction objective on the fixed_random payload; the
# ONLY difference is the center target: WCC uses the per-batch class mean (stateless),
# center loss uses persistent EMA centers (Wen et al. 2016). This sweep trains both arms
# (+ the shared projection-only lambda=0 baseline) at matched lambda and seeds, so the
# table/figure is a clean paired comparison of whether WCC offers benefits
# beyond a general class-compaction objective.
#
# CIFAR-10 IID (alpha->inf), 10 clients, LightSplit-F (fixed_random), L2H-SCH cut, CR=8.
# Each (loss,lambda,seed) cell writes its OWN summary CSV (the --sweep resume key ignores
# wcc_lambda/center_lambda/seed, so a shared CSV would falsely skip cells).
#
# Usage:
#   GPU=0 ./run_center_sensitivity.sh                            # full sweep (~15 runs)
#   GPU=1 LOSSES="center" ./run_center_sensitivity.sh            # center arm only (reuse WCC X1)
#   GPU=0 SEEDS="0" LAMBDAS="0.1" ./run_center_sensitivity.sh    # quick check
# Detach: tmux new -s x8util -d "GPU=0 ./run_center_sensitivity.sh"
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
CR="${CR:-8}"
HEAD="${HEAD:-SCH}"                       # SCH = shared head, PCH = per-client
EPOCHS="${EPOCHS:-100}"
LAMBDAS="${LAMBDAS:-0.01 0.1}"
LOSSES="${LOSSES:-wcc center}"           # arms to run; baseline (lambda=0) always run
SEEDS="${SEEDS:-0 1 2}"
ALPHA="${ALPHA:-10000000}"               # IID
DS=cifar10
PY="${PY:-python}"

case "$HEAD" in
  SCH) HEADFLAG="--deep-head" ;;
  PCH) HEADFLAG="--deep-head --client-heads" ;;
  *)   echo "HEAD must be SCH or PCH"; exit 2 ;;
esac

mkdir -p results/csv logs
rows(){ [ -f "$1" ] && awk 'END{print NR-1}' "$1" 2>/dev/null || echo 0; }

run_cell(){  # tag  extra_flags...
  local tag="$1"; shift
  local csv="results/csv/${tag}.csv"
  if [ "$(rows "$csv")" -ge 1 ]; then echo "[skip] ${tag} (done)"; return; fi
  echo "[$(date +%H:%M:%S)] >>> ${tag}"
  $PY train_split.py --sweep --dataset "$DS" --epochs "$EPOCHS" \
    --num-clients 10 --modes fixed_random --compression-ratio-list "$CR" \
    --alpha-list "$ALPHA" $HEADFLAG --no-viz --gpu "$GPU" "$@" \
    --summary-csv "$csv" --epoch-csv "results/csv/${tag}_epochs.csv" \
    2>&1 | tee "logs/${tag}.log"
}

echo "center-loss utility: $DS L2H-$HEAD cr=${CR} gpu=${GPU} losses=[${LOSSES}] lambdas=[${LAMBDAS}] seeds=[${SEEDS}]"
for s in $SEEDS; do
  # shared projection-only baseline (lambda=0): wcc == center == none
  run_cell "centersens_${DS}_L2H_${HEAD}_cr${CR}_base_w0_s${s}" --wcc-lambda 0 --seed "$s"
  for loss in $LOSSES; do
    for w in $LAMBDAS; do
      case "$loss" in
        wcc)    flag="--wcc-lambda $w" ;;
        center) flag="--center-lambda $w" ;;
        *) echo "unknown loss $loss"; exit 2 ;;
      esac
      run_cell "centersens_${DS}_L2H_${HEAD}_cr${CR}_${loss}_w${w}_s${s}" $flag --seed "$s"
    done
  done
done
echo "[$(date +%H:%M:%S)] DONE  center-loss utility cr${CR} L2H-${HEAD}"
