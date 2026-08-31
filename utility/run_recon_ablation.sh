#!/bin/bash
# Recon Loss Ablation: Accuracy vs recon_lambda across all configurations.
#
# Factors:
#   Mode:      fixed_random (FR)  vs  fixed_random_ll (FR-LL)
#   Deep head: shallow  vs  deep
#   Head type: shared   vs  client (--client-heads)
#   Recon lambda: {0.001, 0.01, 0.1, 1.0}
#   Setting:   IID (alpha=10M) vs Non-IID (alpha=0.1)
#
# Fixed: CIFAR-10, CR=8x (k=512), 10 clients
# Convergence: 2000 max epochs, patience=100 (early stopping on test acc)
#
# Total: 2 modes x 2 depths x 2 heads x 4 lambdas x 2 settings = 64 experiments
# Layout: 4 rounds x 4 GPUs, each GPU runs 4 lambdas sequentially
#   Round 1: FR IID  (16 — already done, will be skipped)
#   Round 2: FR-LL IID  (16)
#   Round 3: FR non-IID  (16)
#   Round 4: FR-LL non-IID  (16)
# Skip logic: experiments already in CSV are skipped automatically.

cd "$(dirname "$0")"

PY="python"
LOGDIR="/tmp/recon_ablation"
CSV="results/csv/recon_ablation_summary.csv"
EPOCH_CSV="results/csv/recon_ablation_epochs.csv"
mkdir -p "$LOGDIR" results/csv

LAMBDAS="0.01 0.1 1.0"
LAMBDA_ARR=(0.01 0.1 1.0)

# ── Colors ──
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
DIM="\033[2m"
RESET="\033[0m"

# ── Config table ──
# tag:label:deep:head:mode:deep_val:client_val
CONFIGS_IID=(
  "fr_shallow_shared:FR:   :Shared:fixed_random:False:False"
  "fr_shallow_client:FR:   :Client:fixed_random:False:True"
  "fr_deep_shared:FR:Deep:Shared:fixed_random:True:False"
  "fr_deep_client:FR:Deep:Client:fixed_random:True:True"
  "frll_shallow_shared:FR-LL:   :Shared:fixed_random_ll:False:False"
  "frll_shallow_client:FR-LL:   :Client:fixed_random_ll:False:True"
  "frll_deep_shared:FR-LL:Deep:Shared:fixed_random_ll:True:False"
  "frll_deep_client:FR-LL:Deep:Client:fixed_random_ll:True:True"
)
CONFIGS_NONIID=(
  "fr_shallow_shared_noniid:FR:   :Shared:fixed_random:False:False"
  "fr_shallow_client_noniid:FR:   :Client:fixed_random:False:True"
  "fr_deep_shared_noniid:FR:Deep:Shared:fixed_random:True:False"
  "fr_deep_client_noniid:FR:Deep:Client:fixed_random:True:True"
  "frll_shallow_shared_noniid:FR-LL:   :Shared:fixed_random_ll:False:False"
  "frll_shallow_client_noniid:FR-LL:   :Client:fixed_random_ll:False:True"
  "frll_deep_shared_noniid:FR-LL:Deep:Shared:fixed_random_ll:True:False"
  "frll_deep_client_noniid:FR-LL:Deep:Client:fixed_random_ll:True:True"
)

# ═══════════════════════════════════════════════════════════════════
# SKIP LOGIC
# ═══════════════════════════════════════════════════════════════════
is_done() {
  local MODE=$1 DEEP_VAL=$2 CLIENT_VAL=$3 LAMBDA=$4 ALPHA=$5
  [ -f "$CSV" ] && awk -F, -v m="$MODE" -v d="$DEEP_VAL" -v c="$CLIENT_VAL" \
    -v l="$LAMBDA" -v a="$ALPHA" '
    NR==1{for(i=1;i<=NF;i++){
      if($i=="mode")mi=i; if($i=="deep_head")di=i; if($i=="client_heads")ci=i
      if($i=="recon_lambda")ri=i; if($i=="alpha")ai=i
    };next}
    $mi==m && $di==d && $ci==c && $ri+0==l+0 && $ai+0==a+0 {found=1;exit}
    END{exit !found}
  ' "$CSV"
}

# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════
print_dashboard() {
  clear
  printf "${BOLD}+--------------------------------------------------------------------------+${RESET}\n"
  printf "${BOLD}|                      RECON LOSS ABLATION DASHBOARD                       |${RESET}\n"
  printf "${BOLD}+--------------------------------------------------------------------------+${RESET}\n"
  printf "  %s    Logs: %s\n\n" "$(date '+%H:%M:%S')" "$LOGDIR"

  # GPU status
  printf "${BOLD}GPUs:${RESET}\n"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx util mu mt; do
    local pct=$((util / 5))
    local bar="" empty=""
    for ((i=0;i<pct;i++)); do bar+="="; done
    for ((i=pct;i<20;i++)); do empty+="."; done
    if [ "$util" -gt 50 ]; then c=$GREEN; else c=$DIM; fi
    printf "  GPU %s [${c}%-20s${RESET}] %3d%%  %5s/%5s MB\n" "$idx" "${bar}${empty}" "$util" "$mu" "$mt"
  done
  echo ""

  local total=0 ndone=0 nrun=0

  for setting_label in "IID (α=10M)" "Non-IID (α=0.1)"; do
    if [[ "$setting_label" == IID* ]]; then
      local cfgs=("${CONFIGS_IID[@]}")
      local alpha="10000000"
    else
      local cfgs=("${CONFIGS_NONIID[@]}")
      local alpha="0.1"
    fi

    printf "\n${BOLD}  ── %s ──${RESET}\n" "$setting_label"
    printf "${BOLD}  %-8s %-5s %-7s" "Method" "Deep" "Head"
    for L in "${LAMBDA_ARR[@]}"; do
      printf " | %12s" "λr=$L"
    done
    printf "${RESET}\n"
    printf "  %.0s-" {1..80}; echo ""

    for cfg in "${cfgs[@]}"; do
      IFS=: read -r tag label deep head mode deep_val client_val <<< "$cfg"
      printf "  %-8s %-5s %-7s" "$label" "$deep" "$head"

      for L in "${LAMBDA_ARR[@]}"; do
        total=$((total+1))
        local logfile="$LOGDIR/${tag}_lambda${L}.log"
        local cell=""

        if [ -f "$CSV" ]; then
          cell=$(awk -F, -v m="$mode" -v d="$deep_val" -v c="$client_val" \
            -v l="$L" -v a="$alpha" '
            NR==1 { for(i=1;i<=NF;i++) {
              if($i=="mode") mi=i; if($i=="best_global_test_acc") ai=i
              if($i=="deep_head") di=i; if($i=="recon_lambda") ri=i
              if($i=="client_heads") ci=i; if($i=="alpha") ali=i
            }; next }
            { if($mi==m && $di==d && $ci==c && $ri+0==l+0 && $ali+0==a+0) {
                printf "%.1f", $ai*100; found=1; exit } }
            END { if(!found) print "" }
          ' "$CSV")
        fi

        if [ -n "$cell" ]; then
          printf " | ${GREEN}%11s%%${RESET}" "$cell"
          ndone=$((ndone+1))
        elif [ -f "$logfile" ]; then
          local last_ep=$(grep -oP 'epoch\s+\K\d+' "$logfile" 2>/dev/null | tail -1)
          local last_acc=$(grep -oP 'best=\K[0-9.]+' "$logfile" 2>/dev/null | tail -1)
          if [ -n "$last_ep" ]; then
            if [ -n "$last_acc" ]; then
              local acc_pct=$(awk "BEGIN{printf \"%.1f\", $last_acc * 100}")
              printf " | ${YELLOW}e%-4s %5s%%${RESET}" "$last_ep" "$acc_pct"
            else
              printf " | ${YELLOW}e%-4s      ${RESET}" "$last_ep"
            fi
            nrun=$((nrun+1))
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
  done

  printf "\n  %.0s-" {1..80}; echo ""
  local pend=$((total - ndone - nrun))
  printf "  ${BOLD}Total:${RESET} %d/%d done, ${YELLOW}%d running${RESET}, ${DIM}%d pending${RESET}\n\n" \
    "$ndone" "$total" "$nrun" "$pend"
}

# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════
run_lambda_sweep() {
  local MODE=$1 DEEP=$2 CLIENT=$3 GPU=$4 TAG=$5 ALPHA=$6

  # Determine bool values for skip check
  local deep_val="False" client_val="False"
  if [[ "$DEEP" == *"--deep-head"* ]]; then deep_val="True"; fi
  if [[ "$CLIENT" == *"--client-heads"* ]]; then client_val="True"; fi

  for L in $LAMBDAS; do
    # Skip if already in CSV
    if is_done "$MODE" "$deep_val" "$client_val" "$L" "$ALPHA"; then
      echo "[GPU $GPU] Skipping $TAG λr=$L (already in CSV)"
      continue
    fi

    PYTHONUNBUFFERED=1 $PY -u train_split.py \
      --mode $MODE $DEEP $CLIENT \
      --recon-lambda $L --alpha $ALPHA --gpu $GPU \
      --dataset cifar10 --num-clients 10 --compression-ratio 8 \
      --epochs 2000 --patience 100 \
      --summary-csv $CSV --epoch-csv $EPOCH_CSV --no-viz \
      >> "$LOGDIR/${TAG}_lambda${L}.log" 2>&1
  done
}

# Dashboard refresh loop
dashboard_loop() {
  while true; do
    print_dashboard
    sleep 15
  done
}

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
echo "=== Recon Loss Ablation: 64 experiments (4 rounds, skip done) ==="
print_dashboard

dashboard_loop &
DASH_PID=$!
trap "kill $DASH_PID 2>/dev/null" EXIT

# Round 1: FR IID (16 — already done, will be skipped)
run_lambda_sweep fixed_random "" "" 0 "fr_shallow_shared" 10000000 &
P1=$!
run_lambda_sweep fixed_random "" "--client-heads" 1 "fr_shallow_client" 10000000 &
P2=$!
run_lambda_sweep fixed_random "--deep-head" "" 2 "fr_deep_shared" 10000000 &
P3=$!
run_lambda_sweep fixed_random "--deep-head" "--client-heads" 3 "fr_deep_client" 10000000 &
P4=$!
wait $P1 $P2 $P3 $P4

# Round 2: FR-LL IID (16)
run_lambda_sweep fixed_random_ll "" "" 0 "frll_shallow_shared" 10000000 &
P1=$!
run_lambda_sweep fixed_random_ll "" "--client-heads" 1 "frll_shallow_client" 10000000 &
P2=$!
run_lambda_sweep fixed_random_ll "--deep-head" "" 2 "frll_deep_shared" 10000000 &
P3=$!
run_lambda_sweep fixed_random_ll "--deep-head" "--client-heads" 3 "frll_deep_client" 10000000 &
P4=$!
wait $P1 $P2 $P3 $P4

# Round 3: FR non-IID (16)
run_lambda_sweep fixed_random "" "" 0 "fr_shallow_shared_noniid" 0.1 &
P1=$!
run_lambda_sweep fixed_random "" "--client-heads" 1 "fr_shallow_client_noniid" 0.1 &
P2=$!
run_lambda_sweep fixed_random "--deep-head" "" 2 "fr_deep_shared_noniid" 0.1 &
P3=$!
run_lambda_sweep fixed_random "--deep-head" "--client-heads" 3 "fr_deep_client_noniid" 0.1 &
P4=$!
wait $P1 $P2 $P3 $P4

# Round 4: FR-LL non-IID (16)
run_lambda_sweep fixed_random_ll "" "" 0 "frll_shallow_shared_noniid" 0.1 &
P1=$!
run_lambda_sweep fixed_random_ll "" "--client-heads" 1 "frll_shallow_client_noniid" 0.1 &
P2=$!
run_lambda_sweep fixed_random_ll "--deep-head" "" 2 "frll_deep_shared_noniid" 0.1 &
P3=$!
run_lambda_sweep fixed_random_ll "--deep-head" "--client-heads" 3 "frll_deep_client_noniid" 0.1 &
P4=$!
wait $P1 $P2 $P3 $P4

# Final
kill $DASH_PID 2>/dev/null
print_dashboard
printf "${GREEN}${BOLD}=== ALL EXPERIMENTS COMPLETE ===${RESET}\n"
echo "Results: $CSV"
