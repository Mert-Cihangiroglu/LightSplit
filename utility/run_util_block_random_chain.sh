#!/usr/bin/env bash
# Run several block_random utility lanes sequentially on one GPU.
# Optionally wait for another tmux session to finish first (so lanes do not
# contend on the same GPU). Lanes are resume-safe, so if a lane is also run on
# another freed GPU meanwhile, this chain just skips its finished cells.
#   WAIT_FOR=ubr_l2h_sch GPU=0 HEADS="L2H_PCH L1H_SCH L1H_PCH" ./run_util_block_random_chain.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
WAIT_FOR="${WAIT_FOR:-}"
HEADS="${HEADS:-L2H_PCH L1H_SCH L1H_PCH}"

if [ -n "$WAIT_FOR" ]; then
  echo "[chain] waiting for tmux '$WAIT_FOR' to end before using GPU $GPU ..."
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 120; done
  echo "[chain] '$WAIT_FOR' finished; starting on GPU $GPU"
fi

for HEAD in $HEADS; do
  echo "[chain $(date '+%F %T')] lane $HEAD on GPU $GPU"
  HEAD="$HEAD" GPU="$GPU" ./run_util_block_random.sh
done
echo "[chain] all lanes done: $HEADS"
