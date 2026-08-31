#!/usr/bin/env bash
# 3-seed non-IID CIFAR-10, L2H-SCH, LightSplit-L (fixed_random_ll, learned lift-back,
# mlp_hidden=512) — the BETTER variant on non-IID (full_sweep: LS-L 44.5/45.4/45.5 vs
# LS-F 45.3/43.9/42.6, flat-to-rising with compression). Averaged for the stable paper
# number, and a reproducibility check vs the paper's single-seed value.
set -uo pipefail
cd "$(dirname "$0")"
GPU="${GPU:-2}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p results/csv logs
echo "LS-L non-IID 3-seed: cifar10 L2H-SCH fixed_random_ll mlp_hidden=512 alpha=0.1 seeds=[$SEEDS] gpu=$GPU"
for S in $SEEDS; do
  tag="noniid_cifar10_L2H_SCH_lsl_s${S}"
  if [ -s "results/csv/${tag}.csv" ] && [ "$(($(wc -l < "results/csv/${tag}.csv")-1))" -ge 3 ]; then
    echo "[$(date +%H:%M:%S)] skip seed $S (done)"; continue; fi
  echo "[$(date +%H:%M:%S)] === LS-L seed $S ==="
  python train_split.py --sweep --dataset cifar10 --epochs 100 --num-clients 10 \
    --modes fixed_random_ll --compression-ratio-list 8 16 32 --alpha-list 0.1 \
    --deep-head --mlp-hidden 512 --seed "$S" --no-viz --gpu "$GPU" \
    --summary-csv "results/csv/${tag}.csv" --epoch-csv "results/csv/${tag}_epochs.csv" \
    2>&1 | tee "logs/${tag}.log"
done
echo "[$(date +%H:%M:%S)] LS-L non-IID 3-seed done."
