#!/usr/bin/env bash
# Figure fig:llm_generality -- GPT-2 (124M/355M/774M/1.5B) fine-tuned on WikiText-103 with the cut
# after round(0.1 * n_layer) blocks; validation-perplexity inflation over Raw for LightSplit-F and
# LightSplit-L at CR 8/16/32.  28 runs (per model: Raw + 2 methods x 3 CRs), one epoch of
# ~20.5M tokens each; about one GPU-hour for gpt2 up to ~10 for gpt2-xl on an RTX A6000.
# Output: $OUT/fig_llm_generality.csv  (model, params_m, mode, cr, cut, best_val_ppl, inflation)
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=fig_llm_generality
cd "$CODE_DIR"
for model in $MODELS; do
  run "$model" raw 1
  for cr in 8 16 32; do run "$model" fixed_random "$cr"; run "$model" fixed_random_ll "$cr"; done
done
"$PY" "$REPRO_DIR/aggregate_llm.py" --root "$OUT" --models $MODELS --out "$OUT/fig_llm_generality.csv"
