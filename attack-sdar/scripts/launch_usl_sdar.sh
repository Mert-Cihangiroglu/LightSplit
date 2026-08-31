#!/usr/bin/env bash
# ============================================================================
# Run inside the TensorFlow (lightsplit-tf) environment.
# Requires the TensorFlow (lightsplit-tf) environment.
#
# Re-runs the block_random SDAR privacy grid in USL (U-shaped SL) to match the
# X1 WCC grid's setting and to enable victim-health gating (VSL can't log the
# victim's label accuracy). Mirrors the existing VSL grid, resume-safe.
#
#   Phase 1 (default): the clean locality dial, lambda=0 only
#       CR{8,16,32} x block{1,2,4} x seeds{0,1,2}  = 27 runs   (~a few hours)
#   Phase 2 (PHASE2=1): add the WCC sweep
#       + WCC{0.001,0.003,0.01,0.03,0.1}           = +135 runs (~a day+)
#     NOTE: at L7/USL the victim DIVERGES often when WCC>0 (master X1 log: some
#     cells 0/4 healthy seeds). Use MORE seeds for the WCC cells and health-gate
#     (victim acc >= 0.40) at aggregation. That's why Phase 2 is opt-in.
#
# Usage on the SDAR server:
#   cd SDAR_SplitNN/scripts
#   bash launch_usl_sdar.sh              # Phase 1 (lambda=0 locality dial)
#   PHASE2=1 SEEDS="0 1 2 3 4" bash launch_usl_sdar.sh   # + WCC sweep, more seeds
#   GPUS="0 1 2 3" LANES_PER_GPU=2 bash launch_usl_sdar.sh
#   DRY=1 bash launch_usl_sdar.sh        # print the plan, launch nothing
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# hard guard: refuse to run without the SDAR env (i.e. on the wrong box)
if [ ! -x python ]; then
  echo "ERROR: python not found."
  echo "       This must run inside the TensorFlow (lightsplit-tf) environment."
  exit 1
fi

export SETTING=usl
export LEVEL="${LEVEL:-7}"
export CRS="${CRS:-8 16 32}"
export BLOCKS="${BLOCKS:-1 2 4}"      # b8 == fixed_random; add "8" if you want the endpoint
export RUNS="${SEEDS:-0 1 2}"
export GPUS="${GPUS:-0 1 2 3}"
export LANES_PER_GPU="${LANES_PER_GPU:-2}"
export DRY="${DRY:-0}"

if [ "${PHASE2:-0}" = "1" ]; then
  export WCCS="${WCCS:-0 0.001 0.003 0.01 0.03 0.1}"
  echo "### USL block_random FULL grid (lambda sweep) -- expect divergence on WCC>0 cells"
else
  export WCCS="0"
  echo "### USL block_random Phase 1 (lambda=0 locality dial)"
fi

echo "setting=usl CRS='$CRS' BLOCKS='$BLOCKS' WCCS='$WCCS' RUNS='$RUNS' GPUS='$GPUS' x$LANES_PER_GPU"
echo
bash run_block_random_grid.sh

echo
echo "When runs finish, aggregate SSIM the same way as VSL:"
echo "  python aggregate_reseed.py --level $LEVEL   # or the block_random aggregator"
echo "topk_sparse USL (optional, separate): SETTING=usl bash run_blackbox_topk_sparse_l7.sh"
