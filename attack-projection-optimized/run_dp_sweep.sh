#!/bin/bash
# DP sweep: noise_multiplier x clip_norm x compression_ratio x deep_head
# Fixed Random LL bottleneck, CIFAR-10, 10 clients, IID
#
# Usage: bash run_dp_sweep.sh --gpu 0

set -e

GPU="${1:---gpu 0}"
COMMON="--dataset cifar10 --epochs 100 --num-clients 10 --alpha 10000000 --mode fixed_random --no-viz $GPU"
SUMMARY_CSV="results/csv/dp_sweep_summary.csv"
EPOCH_CSV="results/csv/dp_sweep_epochs.csv"

echo "DP Sweep: Fixed Random LL + clipping + Gaussian noise"
echo "Summary CSV: $SUMMARY_CSV"
echo "Epoch CSV:   $EPOCH_CSV"
echo ""

for CR in 8 16 32; do
  for SIGMA in 0.0 0.1 0.5 1.0 2.0 5.0; do
    for CLIP in 5.0 10.0; do
      for DEEP in "" "--deep-head"; do
        DTAG=""
        if [ -n "$DEEP" ]; then
          DTAG=" [deep]"
        fi

        echo "================================================================"
        echo "  CR=$CR  sigma=$SIGMA  clip=$CLIP${DTAG}"
        echo "================================================================"

        # Phase 1: Train split model with DP
        python train_split.py $COMMON \
          --compression-ratio $CR \
          --clip-norm $CLIP \
          --noise-multiplier $SIGMA \
          $DEEP \
          --summary-csv "$SUMMARY_CSV" \
          --epoch-csv "$EPOCH_CSV"

        # Phase 2: SDAR attack on the DP model
        python sdar_attack.py \
          --mode fixed_random \
          --compression-ratio $CR \
          --clip-norm $CLIP \
          --noise-multiplier $SIGMA \
          $DEEP \
          --dataset cifar10 \
          --alpha 10000000 \
          --attack-iters 10000 \
          --results-csv results/csv/dp_sdar_attack_results.csv \
          $GPU

        # Phase 3: Privileged attack (Exp 2 oracle + Exp 4 null-space)
        python privileged_attack.py \
          --cr-list $CR \
          --clip-norm $CLIP \
          --noise-multiplier $SIGMA \
          $DEEP \
          --experiments 2 4 \
          --output-dir results/privileged_attack_dp/clip${CLIP}_sigma${SIGMA} \
          $GPU

        echo ""
      done
    done
  done
done

echo "DP sweep complete!"
echo "Results: $SUMMARY_CSV"
