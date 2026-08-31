#!/usr/bin/env bash
# Seed-sequential USL SDAR pipeline. Runs the FULL cell grid for ONE seed to completion
# (so a full single-seed result is available fast), THEN the next seed, per level.
# Order: L7 seed0 -> L7 seed1 -> L4 seed0 -> L4 seed1. Aggregate to mean after seed1.
# Grid per (level,seed): block_random CR{8,16,32} x block{1,2,4} x WCC{0,.001,.003,.01,.03,.1}
#   = 54 runs + topk_sparse CR{8,16,32} = 3 runs.
# Detach: tmux new -s usl_pipe -d "bash run_usl_pipeline.sh".  Resume-safe.
set -uo pipefail
cd "$(dirname "$0")"
log(){ echo "[usl-pipe $(date '+%F %T')] $*"; }

run_stage(){  # level seed
  local L="$1" S="$2"
  log "STAGE L$L seed$S : launch block(54) + topk_sparse(3)"
  GPUS="0 1 2 3" LANES_PER_GPU=2 SETTING=usl LEVEL="$L" CRS="8 16 32" BLOCKS="1 2 4" \
    WCCS="0 0.001 0.003 0.01 0.03 0.1" RUNS="$S" bash run_block_random_grid.sh
  local g=1 cr
  for cr in 8 16 32; do
    tmux new -s "tks_usl_l${L}s${S}_cr${cr}" -d \
      "cd $PWD && SETTING=usl LEVEL=$L GPU=$g RUNS='$S' CRS='$cr' bash run_blackbox_topk_sparse_l7.sh 2>&1 | tee reseed_logs/tks_usl_l${L}s${S}_cr${cr}.log"
    g=$((g+1))
  done
  sleep 40   # let sessions register
  # wait until this stage's block lanes AND its topk_sparse sessions are all gone
  while tmux ls 2>/dev/null | grep -qE "^brgrid_usl_|^tks_usl_l${L}s${S}_"; do sleep 300; done
  log "STAGE L$L seed$S : COMPLETE"
}

for L in 7 4; do
  run_stage "$L" 0
  log "L$L seed0 done -> full single-seed results available. Aggregate: python aggregate_block_usl.py --level $L (seed0)"
  run_stage "$L" 1
  log "L$L seed1 done -> update to MEAN over seeds 0,1."
done
log "PIPELINE COMPLETE (L7+L4, seeds 0,1)."
