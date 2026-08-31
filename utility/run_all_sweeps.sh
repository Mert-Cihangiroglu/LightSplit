#!/usr/bin/env bash
# Master driver: main sweep + 4 one-factor-at-a-time ablations.
# Total runs at default settings:
#   main           : 48 (5 compressed modes x 3 CRs x 3 seeds + raw x 3 seeds)
#   alpha          : 18 (6 modes x 3 alphas, IID baseline excluded)
#   clients        : 18 (6 modes x 3 client counts, c=10 baseline excluded)
#   mal_ratio      : 12 (6 modes x 2 mal values, mal=0.1 baseline excluded)
#   poison_rate    : 24 (6 modes x 4 poison values, poison=0.3 baseline excluded)
#   ----
#   total          : 120 runs
#
# Each run: 30 epochs, ~25 min on a single A100 -> ~50 hrs total wall time.
#
# Usage: bash run_all_sweeps.sh
set -e

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-results/backdoor}"
PYTHON="${PYTHON:-python}"

export GPU EPOCHS NUM_WORKERS BATCH_SIZE OUT_DIR PYTHON

echo "############################################################"
echo "# main sweep (mode x CR x seed)"
echo "############################################################"
bash run_backdoor_sweep.sh

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
echo "# All sweeps complete."
echo "############################################################"
ls -1d "$OUT_DIR"/cifar10_* | wc -l
echo "Run directories above. See $OUT_DIR/sweep.log for combined output."
