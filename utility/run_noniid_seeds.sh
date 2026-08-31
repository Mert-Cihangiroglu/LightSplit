#!/usr/bin/env bash
# 3-seed non-IID CIFAR-10, L2H-SCH, LightSplit-F (fixed_random, lambda_WCC=0,
# alpha=0.1, CR{8,16,32}) — averaged for a stable paper number, since a single
# non-IID seed wobbles under heavy label skew.
set -uo pipefail
cd "$(dirname "$0")"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p results/csv logs
echo "non-IID 3-seed: cifar10 L2H-SCH fixed_random alpha=0.1 seeds=[$SEEDS] gpu=$GPU"
for S in $SEEDS; do
  tag="noniid_cifar10_L2H_SCH_fr_s${S}"
  if [ -s "results/csv/${tag}.csv" ] && [ "$(($(wc -l < "results/csv/${tag}.csv")-1))" -ge 3 ]; then
    echo "[$(date +%H:%M:%S)] skip seed $S (done)"; continue
  fi
  echo "[$(date +%H:%M:%S)] === seed $S ==="
  python train_split.py --sweep --dataset cifar10 --epochs 100 --num-clients 10 \
    --modes fixed_random --compression-ratio-list 8 16 32 --alpha-list 0.1 \
    --deep-head --seed "$S" --no-viz --gpu "$GPU" \
    --summary-csv "results/csv/${tag}.csv" --epoch-csv "results/csv/${tag}_epochs.csv" \
    2>&1 | tee "logs/${tag}.log"
done
echo "[$(date +%H:%M:%S)] non-IID 3-seed done."
