#!/usr/bin/env bash
# Supervisor: wait for the L7 USL grid to finish, then launch the SAME full grid at L4.
# L4 = 8192-d cut (k=1024/512/256), ~3x slower than L7. Sequential (GPUs already saturated
# at 2 lanes/GPU). Run detached: tmux new -s usl_super -d "bash run_usl_l4_after_l7.sh".
set -uo pipefail
cd "$(dirname "$0")"
log(){ echo "[usl-super $(date '+%F %T')] $*"; }
log "waiting for L7 grid (brgrid_usl_* / tks_usl_cr*) to finish..."
while tmux ls 2>/dev/null | grep -qE '^brgrid_usl_|^tks_usl_cr'; do sleep 300; done
log "L7 done -> launching L4 full grid (108 block + 6 topk_sparse, 2 seeds)."
GPUS="0 1 2 3" LANES_PER_GPU=2 SETTING=usl LEVEL=4 CRS="8 16 32" BLOCKS="1 2 4" \
  WCCS="0 0.001 0.003 0.01 0.03 0.1" RUNS="0 1" bash run_block_random_grid.sh
g=1
for cr in 8 16 32; do
  tmux new -s "tks_usl_l4_cr${cr}" -d "cd $PWD && SETTING=usl LEVEL=4 GPU=$g RUNS='0 1' CRS='$cr' bash run_blackbox_topk_sparse_l7.sh 2>&1 | tee reseed_logs/tks_usl_l4_cr${cr}.log"
  g=$((g+1))
done
log "L4 launched. Aggregate per level when done."
