#!/usr/bin/env bash
# HAR DP probe cells of tab:privacy_all (Raw+DP and LightSplit-F+DP).
# Trains the two arms with per-sample L2 clipping + Gaussian noise on the
# transmitted payload (sigma 0.02; clip = median payload norm, from
# measure_har_dp_norms.py) at 3 seeds, runs the probes, and reports the
# median-over-seeds R^2 / re-identification as percent change vs the Raw row
# of fig_har.csv (run fig_har.sh first). 6 runs, minutes each on CPU.
set -euo pipefail
source "$(dirname "$0")/common.sh"; FIG=har_dp
SIGMA="${SIGMA:-0.02}"
CLIP_RAW="${CLIP_RAW:-8.074}"
CLIP_FIX="${CLIP_FIX:-3.555}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p "$OUT/$FIG"
cd "$CODE_DIR"
for s in $SEEDS; do
  for arm in none fixed; do
    case "$arm" in
      none)  extra=(--dp_clip "$CLIP_RAW"); tag="none_w0_dp_s$s" ;;
      fixed) extra=(--liftback fixed --k 32 --dp_clip "$CLIP_FIX"); tag="fixed_w0_dp_s$s" ;;
    esac
    d="$OUT/$FIG/$tag"
    [ -s "$d/attacks.json" ] && { echo "[skip] $tag"; continue; }
    echo "[$(date +%H:%M:%S)] $tag"
    "$PY" train_har.py ${extra[@]+"${extra[@]}"} --dp_sigma "$SIGMA" --seed "$s" --out "$d"
    "$PY" attacks.py --run "$d" --attack all
  done
done
"$PY" "$REPRO_DIR/aggregate_har_dp.py" --dp "$OUT/$FIG" --fig-har "$OUT/fig_har.csv" \
  --out "$OUT/tab_har_dp.csv"
