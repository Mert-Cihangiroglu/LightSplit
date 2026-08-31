#!/usr/bin/env bash
# Resume from alpha ablation onwards (main sweep already complete).
set -e

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-results/backdoor}"
PYTHON="${PYTHON:-python}"

export GPU EPOCHS NUM_WORKERS BATCH_SIZE OUT_DIR PYTHON

echo "############################################################"
echo "# alpha ablation"
echo "############################################################"
bash run_ablation_alpha.sh

echo "############################################################"
echo "# client-count ablation"
echo "############################################################"
bash run_ablation_clients.sh

echo "############################################################"
echo "# malicious-ratio ablation"
echo "############################################################"
bash run_ablation_mal_ratio.sh

echo "############################################################"
echo "# poison-rate ablation"
echo "############################################################"
bash run_ablation_poison_rate.sh

echo "############################################################"
echo "# All ablations complete."
echo "############################################################"
ls -1d "$OUT_DIR"/cifar10_* | wc -l
