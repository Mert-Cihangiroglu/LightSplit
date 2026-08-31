#!/usr/bin/env bash
# Table tab:har_noniid -- UCI HAR under Dirichlet label heterogeneity, 5 clients, alpha in {0.5, 0.3, 0.1}.
# Arms: Raw, top-k sparse, Learned 1x1 (codec), LightSplit-F and LightSplit-L each with lambda_WCC in {0, 0.1},
# at k in {32, 16, 8} (CR 8/16/32), seeds 0 1 2, 200 epochs; 171 runs of a few minutes each on one GPU.
# The alpha -> infinity column comes from the single-client IID sweep (fig_har.sh); pass its CSV in IID_CSV
# (default: $OUT/fig_har.csv if present). At alpha 0.1 seed 2 is left out of every arm (EXCLUDE), as in the paper.
# Output: $OUT/tab_har_noniid.csv
set -uo pipefail
_EPOCHS="${EPOCHS:-}"; source "$(dirname "$0")/common.sh"; TAB=tab_har_noniid; RUN_DIR="$OUT/$TAB"; mkdir -p "$RUN_DIR"
ALPHAS="${ALPHAS:-0.5 0.3 0.1}"; KS="${KS:-32 16 8}"; EPOCHS="${_EPOCHS:-200}"; CLIENTS="${CLIENTS:-5}"
METHODS="${METHODS:-none:0 topk_sparse:0 codec:0 fixed:0 learned:0 fixed:0.1 learned:0.1}"; EXCLUDE="${EXCLUDE:-0.1:2}"
IID_CSV="${IID_CSV:-$OUT/fig_har.csv}"
if [ "${SMOKE:-0}" = 1 ]; then ALPHAS=0.5; SEEDS=0; KS=32; EPOCHS=2; METHODS="none:0 fixed:0"; fi
for m in $METHODS; do lb="${m%%:*}"; w="${m##*:}"
  ks="$KS"; [ "$lb" = none ] && ks=32
  for k in $ks; do for al in $ALPHAS; do for s in $SEEDS; do
    tag="${lb}_w${w}_k${k}_a${al}_s${s}"
    [ -s "$RUN_DIR/$tag/summary.json" ] && { echo "[skip] $tag"; continue; }
    echo "[$(date +%H:%M:%S)] $tag"
    "$PY" "$CODE_DIR/train_noniid.py" --liftback "$lb" --wcc_lambda "$w" --k "$k" --d 256 --alpha "$al" --num_clients "$CLIENTS" \
        --epochs "$EPOCHS" --seed "$s" "${DATA[@]}" --out "$RUN_DIR/$tag" > "$RUN_DIR/$tag.log" 2>&1 || echo "[fail] $tag (see $RUN_DIR/$tag.log)"
  done; done; done
done
iid=(); [ -s "$IID_CSV" ] && iid=(--iid-csv "$IID_CSV")
excl=(); for e in $EXCLUDE; do excl+=(--exclude "$e"); done
"$PY" "$REPRO_DIR/aggregate_har_noniid.py" --results "$RUN_DIR" "${iid[@]}" "${excl[@]}" --out "$OUT/$TAB.csv"
