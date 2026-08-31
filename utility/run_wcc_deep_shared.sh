#!/bin/bash
# WCC accuracy sweep: FR deep shared + FR-LL deep shared
# WCC lambda: {0.001, 0.01, 0.1, 1.0, 10.0}  (no 0 — baselines already exist)
# IID + non-IID
#
# Fixed: CIFAR-10, CR=8x, 10 clients, patience=100
# Total: 2 modes x 5 lambdas x 2 alphas = 20 experiments, 4 GPUs

set -e
cd "$(dirname "$0")"

PY="python"
LOGDIR="/tmp/wcc_deep_shared"
CSV="results/csv/wcc_deep_shared_summary.csv"
EPOCH_CSV="results/csv/wcc_deep_shared_epochs.csv"
mkdir -p "$LOGDIR" results/csv

LAMBDAS=(0.001 0.01 0.1 1.0 10.0)
COMMON_BASE="--dataset cifar10 --num-clients 10 \
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
  printf "${BOLD}+----------------------------------------------------------------------+${RESET}\n"
  printf "${BOLD}|         WCC ACCURACY: FR deep + FR-LL deep (shared)                  |${RESET}\n"
  printf "${BOLD}+----------------------------------------------------------------------+${RESET}\n"
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

  for alpha_info in "10000000:IID" "0.1:non-IID"; do
    IFS=: read -r alpha alpha_label <<< "$alpha_info"

    printf "${BOLD}%s (alpha=%s):${RESET}\n" "$alpha_label" "$alpha"
    printf "${BOLD}%-8s  %12s %12s %12s %12s %12s${RESET}\n" \
      "Method" "lw=0.001" "lw=0.01" "lw=0.1" "lw=1.0" "lw=10.0"
    printf "%.0s-" {1..72}; echo ""

    for row in "fr_deep:FR:fixed_random" "frll_deep:FR-LL:fixed_random_ll"; do
      IFS=: read -r tag label mode <<< "$row"
      printf "%-8s" "$label"
      for L in "${LAMBDAS[@]}"; do
        local logfile="$LOGDIR/${tag}_a${alpha}_wcc${L}.log"
        local cell=""

        if [ -f "$CSV" ]; then
          cell=$(awk -F, -v m="$mode" -v wl="$L" -v a="$alpha" '
            NR==1{for(i=1;i<=NF;i++){if($i=="mode")mi=i;if($i=="best_global_test_acc")ai=i;if($i=="wcc_lambda")wi=i;if($i=="alpha")ali=i};next}
            {if($mi==m&&$wi+0==wl+0&&$ali+0==a+0){printf "%.1f",$ai*100;found=1;exit}}
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
    printf "%.0s-" {1..72}; echo ""
    echo ""
  done
}

run_sweep() {
  local MODE=$1 GPU=$2 TAG=$3 ALPHA=$4
  local COMMON="$COMMON_BASE --alpha $ALPHA"
  for L in "${LAMBDAS[@]}"; do
    # Skip if result already in CSV
    if [ -f "$CSV" ]; then
      local exists=$(awk -F, -v m="$MODE" -v wl="$L" -v a="$ALPHA" '
        NR==1{for(i=1;i<=NF;i++){if($i=="mode")mi=i;if($i=="wcc_lambda")wi=i;if($i=="alpha")ali=i};next}
        {if($mi==m&&$wi+0==wl+0&&$ali+0==a+0){print "yes";exit}}' "$CSV")
      if [ "$exists" = "yes" ]; then
        echo "[GPU $GPU] Skipping $MODE alpha=$ALPHA wcc=$L (already in CSV)"
        continue
      fi
    fi
    PYTHONUNBUFFERED=1 $PY -u train_split.py \
      --mode $MODE --wcc-lambda $L --gpu $GPU \
      $COMMON \
      >> "$LOGDIR/${TAG}_a${ALPHA}_wcc${L}.log" 2>&1
  done
}

dashboard_loop() {
  while true; do print_dashboard; sleep 15; done
}

echo "=== WCC Accuracy: 20 experiments (IID + non-IID) ==="
print_dashboard

dashboard_loop &
DASH_PID=$!
trap "kill $DASH_PID 2>/dev/null" EXIT

# Round 1: IID (alpha=10000000)
# GPU 0: FR all lambdas, GPU 1: FR-LL all lambdas
run_sweep fixed_random    0 "fr_deep"   10000000 &
run_sweep fixed_random_ll 1 "frll_deep" 10000000 &
# Round 1 also: non-IID on GPU 2-3
run_sweep fixed_random    2 "fr_deep"   0.1 &
run_sweep fixed_random_ll 3 "frll_deep" 0.1 &
wait

kill $DASH_PID 2>/dev/null
print_dashboard
printf "${GREEN}${BOLD}=== ALL 20 EXPERIMENTS COMPLETE ===${RESET}\n"
echo "Results: $CSV"
