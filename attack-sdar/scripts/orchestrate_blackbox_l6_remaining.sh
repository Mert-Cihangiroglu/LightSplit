#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LEVEL="${LEVEL:-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SECONDS="${POLL_SECONDS:-300}"

timestamp() {
  date '+%F %T'
}

wait_for_queue() {
  local logfile="$1"
  local marker="$2"

  echo "[$(timestamp)] Waiting for ${logfile}: ${marker}"
  while true; do
    if [[ -f "$logfile" ]] && grep -q "$marker" "$logfile"; then
      echo "[$(timestamp)] Complete: ${logfile}"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

wait_for_phase() {
  local prefix="$1"
  wait_for_queue "${prefix}_gpu1_launcher.out" "Queue gpu1 complete"
  wait_for_queue "${prefix}_gpu2_launcher.out" "Queue gpu2 complete"
  wait_for_queue "${prefix}_gpu3_launcher.out" "Queue gpu3 complete"
}

launch_bottleneck_phase() {
  local mode="$1"
  local prefix="$2"

  echo "[$(timestamp)] Launching ${mode} level-${LEVEL} queues."
  setsid nohup ./run_blackbox_bottleneck_level_queue.sh "$mode" "$LEVEL" gpu1 > "${prefix}_gpu1_launcher.out" 2>&1 &
  setsid nohup ./run_blackbox_bottleneck_level_queue.sh "$mode" "$LEVEL" gpu2 > "${prefix}_gpu2_launcher.out" 2>&1 &
  setsid nohup ./run_blackbox_bottleneck_level_queue.sh "$mode" "$LEVEL" gpu3 > "${prefix}_gpu3_launcher.out" 2>&1 &
}

echo "=== SDAR level-${LEVEL} remaining-grid orchestrator ==="
echo "python=${PYTHON_BIN}"
echo "poll_seconds=${POLL_SECONDS}"

wait_for_phase "blackbox_l${LEVEL}_fixed"

launch_bottleneck_phase learned_1x1 "blackbox_l${LEVEL}_learned"
wait_for_phase "blackbox_l${LEVEL}_learned"

echo "[$(timestamp)] All level-${LEVEL} fixed-random and learned-1x1 queues are complete."
