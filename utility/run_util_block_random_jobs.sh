#!/usr/bin/env bash
# Run a list of block_random utility lanes (dataset:head) sequentially on one GPU.
# Optionally wait for another tmux session to finish first. Resume-safe.
#   WAIT_FOR=ubr_g0 GPU=0 JOBS="cifar100:L1H_SCH gtsrb:L1H_SCH fmnist:L1H_SCH" ./run_util_block_random_jobs.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
WAIT_FOR="${WAIT_FOR:-}"
: "${JOBS:?need JOBS='dataset:head dataset:head ...'}"

if [ -n "$WAIT_FOR" ]; then
  echo "[jobs] waiting for tmux '$WAIT_FOR' before using GPU $GPU ..."
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 120; done
fi

for J in $JOBS; do
  DS="${J%%:*}"; HD="${J##*:}"
  echo "[jobs $(date '+%F %T')] $DS / $HD on GPU $GPU"
  DATASET="$DS" HEAD="$HD" GPU="$GPU" ./run_util_block_random.sh
done
echo "[jobs] done: $JOBS"
