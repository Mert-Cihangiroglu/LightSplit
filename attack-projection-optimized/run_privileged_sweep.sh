#!/bin/bash
# Populate privileged attack results for ALL Fixed Random LL configs in sdar-privacy.tex
# Configs: shallow/deep x recon_lambda=0/0.1 x CR=8/16/32
#
# Usage: bash run_privileged_sweep.sh --gpu 0

set -e

GPU="${1:---gpu 0}"
OUTBASE="results/privileged_attack"

echo "Privileged attack sweep: all Fixed Random LL configs"
echo ""

for CR in 8 16 32; do
  for DEEP in "" "--deep-head"; do
    for RL in 0.0 0.1; do
      DTAG=""
      DARG=""
      if [ -n "$DEEP" ]; then
        DTAG="_deep"
        DARG="--deep-head"
      fi
      RLTAG=""
      if [ "$RL" != "0.0" ]; then
        RLTAG="_rl${RL}"
      fi

      OUTDIR="${OUTBASE}/fr_CR${CR}${DTAG}${RLTAG}"

      echo "================================================================"
      echo "  CR=$CR  deep=${DEEP:-no}  recon_lambda=$RL"
      echo "  output: $OUTDIR"
      echo "================================================================"

      python privileged_attack.py \
        --cr-list $CR \
        $DARG \
        --recon-lambda $RL \
        --experiments 0 1 2 3 4 \
        --output-dir "$OUTDIR" \
        $GPU

      echo ""
    done
  done
done

echo "Privileged sweep complete!"
