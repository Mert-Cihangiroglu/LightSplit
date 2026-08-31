#!/bin/bash
# WCC Ablation: Accuracy vs WCC strength across all configurations.
#
# Factors:
#   Mode:      fixed_random (FR)  vs  fixed_random_ll (FR-LL)
#   Deep head: shallow  vs  deep
#   Head type: shared   vs  client (--client-heads)
#   WCC lambda: {0, 0.001, 0.01, 0.1, 1.0}
#
# Fixed: CIFAR-10 IID, CR=8x (k=512), 10 clients
# Convergence: 2000 max epochs, patience=100 (early stopping on test acc)
#
# Total: 2 modes x 2 depths x 2 heads x 5 lambdas = 40 experiments
# Layout: 2 rounds x 4 GPUs, each GPU runs 5 lambdas sequentially

cd "$(dirname "$0")"

PY="python"
LOGDIR="/tmp/wcc_ablation"
CSV="results/csv/wcc_ablation_summary.csv"
EPOCH_CSV="results/csv/wcc_ablation_epochs.csv"
mkdir -p "$LOGDIR" results/csv

LAMBDAS="0 0.001 0.01 0.1 1.0"
LAMBDA_ARR=(0 0.001 0.01 0.1 1.0)
COMMON="--dataset cifar10 --alpha 10000000 --num-clients 10 \
  --compression-ratio 8 --epochs 2000 --patience 100 \
  --summary-csv $CSV --epoch-csv $EPOCH_CSV --no-viz"

# ── Colors ──
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
DIM="\033[2m"
RESET="\033[0m"

# ── Config table (tag:label:deep:head:mode:deep_val:client_val) ──
CONFIGS=(
  "fr_shallow_shared:FR:   :Shared:fixed_random:False:False"
  "fr_shallow_client:FR:   :Client:fixed_random:False:True"
  "fr_deep_shared:FR:Deep:Shared:fixed_random:True:False"
  "fr_deep_client:FR:Deep:Client:fixed_random:True:True"
  "frll_shallow_shared:FR-LL:   :Shared:fixed_random_ll:False:False"
  "frll_shallow_client:FR-LL:   :Client:fixed_random_ll:False:True"
  "frll_deep_shared:FR-LL:Deep:Shared:fixed_random_ll:True:False"
  "frll_deep_client:FR-LL:Deep:Client:fixed_random_ll:True:True"
)

# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════
print_dashboard() {
  clear
  printf "${BOLD}+--------------------------------------------------------------------------+${RESET}\n"
  printf "${BOLD}|                        WCC ABLATION DASHBOARD                            |${RESET}\n"
  printf "${BOLD}+--------------------------------------------------------------------------+${RESET}\n"
  printf "  %s    Logs: %s\n\n" "$(date '+%H:%M:%S')" "$LOGDIR"

  # GPU status
  printf "${BOLD}GPUs:${RESET}\n"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx util mu mt; do
    local pct=$((util / 5))
    local bar=""
    local empty=""
    for ((i=0;i<pct;i++)); do bar+="="; done
    for ((i=pct;i<20;i++)); do empty+="."; done
    if [ "$util" -gt 50 ]; then c=$GREEN; else c=$DIM; fi
    printf "  GPU %s [${c}%-20s${RESET}] %3d%%  %5s/%5s MB\n" "$idx" "${bar}${empty}" "$util" "$mu" "$mt"
  done
  echo ""

  # Table header
  printf "${BOLD}%-8s %-5s %-7s" "Method" "Deep" "Head"
  for L in "${LAMBDA_ARR[@]}"; do
    printf " | %12s" "l=$L"
  done
  printf "${RESET}\n"
  printf "%.0s-" {1..90}; echo ""

  # Table rows
  local total=0 ndone=0 nrun=0
  for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r tag label deep head mode deep_val client_val <<< "$cfg"
    printf "%-8s %-5s %-7s" "$label" "$deep" "$head"

    for L in "${LAMBDA_ARR[@]}"; do
      total=$((total+1))
      local logfile="$LOGDIR/${tag}_lambda${L}.log"
      local cell=""

      # Check CSV for final result
      if [ -f "$CSV" ]; then
        cell=$(awk -F, -v m="$mode" -v d="$deep_val" -v c="$client_val" -v l="$L" '
          NR==1 { for(i=1;i<=NF;i++) {
            if($i=="mode") mi=i
            if($i=="best_global_test_acc") ai=i
            if($i=="deep_head") di=i
            if($i=="wcc_lambda") wi=i
            if($i=="client_heads") ci=i
          }; next }
          { if($mi==m && $di==d && $ci==c && $wi+0==l+0) { printf "%.1f", $ai*100; found=1; exit } }
          END { if(!found) print "" }
        ' "$CSV")
      fi

      if [ -n "$cell" ]; then
        printf " | ${GREEN}%11s%%${RESET}" "$cell"
        ndone=$((ndone+1))
      elif [ -f "$logfile" ]; then
        # Parse live epoch + accuracy
        local last=$(grep "global test acc" "$logfile" 2>/dev/null | tail -1)
        if [ -n "$last" ]; then
          local ep=$(echo "$last" | grep -oP 'epoch \K\d+')
          local acc=$(echo "$last" | grep -oP 'global test acc\s+\K[0-9.]+')
          printf " | ${YELLOW}e%-4s %5s%%${RESET}" "$ep" "$acc"
          nrun=$((nrun+1))
        elif grep -q "best global test acc" "$logfile" 2>/dev/null; then
          printf " | ${GREEN}%12s${RESET}" "done"
          ndone=$((ndone+1))
        else
          printf " | ${CYAN}%12s${RESET}" "loading..."
          nrun=$((nrun+1))
        fi
      else
        printf " | ${DIM}%12s${RESET}" "---"
      fi
    done
    echo ""
  done

  printf "%.0s-" {1..90}; echo ""
  local pend=$((total - ndone - nrun))
  printf "${BOLD}Total:${RESET} %d/%d done, ${YELLOW}%d running${RESET}, ${DIM}%d pending${RESET}\n\n" \
    "$ndone" "$total" "$nrun" "$pend"
}

# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════
run_lambda_sweep() {
  local MODE=$1 DEEP=$2 CLIENT=$3 GPU=$4 TAG=$5
  for L in $LAMBDAS; do
    # Skip if result already in CSV
    if [ -f "$CSV" ]; then
      # Determine deep_val and client_val for CSV lookup
      local dv="False" cv="False"
      if [[ "$DEEP" == *deep-head* ]]; then dv="True"; fi
      if [[ "$CLIENT" == *client-heads* ]]; then cv="True"; fi
      local exists=$(awk -F, -v m="$MODE" -v d="$dv" -v c="$cv" -v wl="$L" '
        NR==1{for(i=1;i<=NF;i++){if($i=="mode")mi=i;if($i=="deep_head")di=i;if($i=="client_heads")ci=i;if($i=="wcc_lambda")wi=i};next}
        {if($mi==m&&$di==d&&$ci==c&&$wi+0==wl+0){print "yes";exit}}' "$CSV")
      if [ "$exists" = "yes" ]; then
        echo "[GPU $GPU] Skipping $TAG wcc=$L (already in CSV)"
        continue
      fi
    fi
    PYTHONUNBUFFERED=1 $PY -u train_split.py \
      --mode $MODE $DEEP $CLIENT \
      --wcc-lambda $L --gpu $GPU \
      $COMMON \
      >> "$LOGDIR/${TAG}_lambda${L}.log" 2>&1
  done
}

# Dashboard refresh loop (runs in background)
dashboard_loop() {
  while true; do
    print_dashboard
    sleep 15
  done
}

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
echo "=== WCC Ablation: 40 experiments ==="
print_dashboard

# Start dashboard refresh in background
dashboard_loop &
DASH_PID=$!
trap "kill $DASH_PID 2>/dev/null" EXIT

# Round 1: FR
run_lambda_sweep fixed_random "" "" 0 "fr_shallow_shared" &
P1=$!
run_lambda_sweep fixed_random "" "--client-heads" 1 "fr_shallow_client" &
P2=$!
run_lambda_sweep fixed_random "--deep-head" "" 2 "fr_deep_shared" &
P3=$!
run_lambda_sweep fixed_random "--deep-head" "--client-heads" 3 "fr_deep_client" &
P4=$!
wait $P1 $P2 $P3 $P4

# Round 2: FR-LL
run_lambda_sweep fixed_random_ll "" "" 0 "frll_shallow_shared" &
P1=$!
run_lambda_sweep fixed_random_ll "" "--client-heads" 1 "frll_shallow_client" &
P2=$!
run_lambda_sweep fixed_random_ll "--deep-head" "" 2 "frll_deep_shared" &
P3=$!
run_lambda_sweep fixed_random_ll "--deep-head" "--client-heads" 3 "frll_deep_client" &
P4=$!
wait $P1 $P2 $P3 $P4

# Final dashboard
kill $DASH_PID 2>/dev/null
print_dashboard
printf "${GREEN}${BOLD}=== ALL 40 EXPERIMENTS COMPLETE ===${RESET}\n"
echo "Results: $CSV"
