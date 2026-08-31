#!/bin/bash
# Recon loss accuracy sweep: FR deep shared + FR-LL deep shared
# Recon lambda: {0.001, 0.01, 0.1, 1.0}  (no 0 — baselines from WCC sweep)
#
# Fixed: CIFAR-10 IID, CR=8x, 10 clients, patience=100
# Total: 2 modes x 4 lambdas = 8 experiments, 2 GPUs

set -e
cd "$(dirname "$0")"

CONDA="conda"
ENV="LightSplit"
LOGDIR="/tmp/recon_deep_shared"
CSV="results/csv/recon_deep_shared_summary.csv"
EPOCH_CSV="results/csv/recon_deep_shared_epochs.csv"
mkdir -p "$LOGDIR" results/csv

LAMBDAS=(0.001 0.01 0.1 1.0)
COMMON="--dataset cifar10 --alpha 10000000 --num-clients 10 \
  --compression-ratio 8 --epochs 2000 --patience 100 \
  --deep-head --summary-csv $CSV --epoch-csv $EPOCH_CSV --no-viz"

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
DIM="\033[2m"
RESET="\033[0m"

print_dashboard() {
  clear
  printf "${BOLD}+------------------------------------------------------+${RESET}\n"
  printf "${BOLD}|   RECON ACCURACY: FR deep + FR-LL deep (shared)      |${RESET}\n"
  printf "${BOLD}+------------------------------------------------------+${RESET}\n"
  printf "  %s\n\n" "$(date '+%H:%M:%S')"

  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx util mu mt; do
    local pct=$((util / 5)) bar="" empty=""
    for ((i=0;i<pct;i++)); do bar+="="; done
    for ((i=pct;i<20;i++)); do empty+="."; done
    if [ "$util" -gt 50 ]; then c=$GREEN; else c=$DIM; fi
    printf "  GPU %s [${c}%-20s${RESET}] %3d%%  %5s/%5s MB\n" "$idx" "${bar}${empty}" "$util" "$mu" "$mt"
  done
  echo ""

  printf "${BOLD}%-8s  %12s %12s %12s %12s${RESET}\n" \
    "Method" "lr=0.001" "lr=0.01" "lr=0.1" "lr=1.0"
  printf "%.0s-" {1..58}; echo ""

  for row in "fr_deep:FR:fixed_random" "frll_deep:FR-LL:fixed_random_ll"; do
    IFS=: read -r tag label mode <<< "$row"
    printf "%-8s" "$label"
    for L in "${LAMBDAS[@]}"; do
      local logfile="$LOGDIR/${tag}_rl${L}.log"
      local cell=""

      # Check CSV for final result
      if [ -f "$CSV" ]; then
        cell=$(awk -F, -v m="$mode" -v rl="$L" '
          NR==1{for(i=1;i<=NF;i++){if($i=="mode")mi=i;if($i=="best_global_test_acc")ai=i;if($i=="recon_lambda")ri=i};next}
          {if($mi==m&&$ri+0==rl+0){printf "%.1f",$ai*100;found=1;exit}}
          END{if(!found)print ""}' "$CSV")
      fi

      if [ -n "$cell" ]; then
        printf "  ${GREEN}%11s%%${RESET}" "$cell"
      elif [ -f "$logfile" ]; then
        local last=$(grep "global test acc" "$logfile" 2>/dev/null | tail -1)
        if [ -n "$last" ]; then
          local ep=$(echo "$last" | grep -oP 'epoch \K\d+')
          local acc=$(echo "$last" | grep -oP 'global test acc\s+\K[0-9.]+')
          printf "  ${YELLOW}e%-4s %5s%%${RESET}" "$ep" "$acc"
        elif grep -q "best global test acc" "$logfile" 2>/dev/null; then
          printf "  ${GREEN}%12s${RESET}" "done"
        else
          printf "  ${CYAN}%12s${RESET}" "loading..."
        fi
      else
        printf "  ${DIM}%12s${RESET}" "---"
      fi
    done
    echo ""
  done
  printf "%.0s-" {1..58}; echo ""
  echo ""
}

run_sweep() {
  local MODE=$1 GPU=$2 TAG=$3
  for L in "${LAMBDAS[@]}"; do
    PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n $ENV python -u train_split.py \
      --mode $MODE --recon-lambda $L --gpu $GPU \
      $COMMON \
      >> "$LOGDIR/${TAG}_rl${L}.log" 2>&1
  done
}

dashboard_loop() {
  while true; do print_dashboard; sleep 15; done
}

echo "=== Recon Accuracy: FR deep + FR-LL deep (8 experiments) ==="
print_dashboard

dashboard_loop &
DASH_PID=$!
trap "kill $DASH_PID 2>/dev/null" EXIT

# 4 GPUs: split lambdas across GPUs, both modes in parallel
# GPU 0: FR  lr=0.001,0.01   GPU 1: FR  lr=0.1,1.0
# GPU 2: FR-LL lr=0.001,0.01 GPU 3: FR-LL lr=0.1,1.0
run_pair() {
  local MODE=$1 GPU=$2 TAG=$3 L1=$4 L2=$5
  PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n $ENV python -u train_split.py \
    --mode $MODE --recon-lambda $L1 --gpu $GPU $COMMON \
    >> "$LOGDIR/${TAG}_rl${L1}.log" 2>&1
  PYTHONUNBUFFERED=1 $CONDA run --no-capture-output -n $ENV python -u train_split.py \
    --mode $MODE --recon-lambda $L2 --gpu $GPU $COMMON \
    >> "$LOGDIR/${TAG}_rl${L2}.log" 2>&1
}

run_pair fixed_random   0 "fr_deep"   0.001 0.01 &
run_pair fixed_random   1 "fr_deep"   0.1   1.0  &
run_pair fixed_random_ll 2 "frll_deep" 0.001 0.01 &
run_pair fixed_random_ll 3 "frll_deep" 0.1   1.0  &
wait

kill $DASH_PID 2>/dev/null
print_dashboard
printf "${GREEN}${BOLD}=== ALL 8 EXPERIMENTS COMPLETE ===${RESET}\n"
echo "Results: $CSV"
