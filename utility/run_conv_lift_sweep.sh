#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
#
# Fixed Random Conv sweep for fixed_random_conv mode.
# CIFAR-10, client-heads, 10 clients, 100 epochs.
# Sweep: CR ∈ {8} × KS ∈ {1, 3, 5, 9} × alpha ∈ {10000000} = 4 runs (deep decompress)
#        + CR ∈ {8} × KS ∈ {1} × alpha ∈ {10000000} = 1 run (baseline, no deep decompress)
#
# Usage:
#   bash run_conv_lift_sweep.sh --gpus 1       # run on GPU 1
#   bash run_conv_lift_sweep.sh --gpus 1,2     # parallel on GPUs 1,2
#   bash run_conv_lift_sweep.sh --dry-run

set -euo pipefail

CONDA_ENV="LightSplit"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/train_split.py"
DATASET="cifar10"
EPOCHS=100
BATCH_SIZE=128
NUM_CLIENTS=10
SEED=42
MODE="fixed_random_conv"
GPUS=(1)

CRS=(8)
ALPHAS=(10000000)
KERNEL_SIZES=(1 3 5 9)
DECOMPRESS_LR="3e-3"
RECON_LAMBDA="0.1"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/fixed-random-conv-results"
SUMMARY_CSV="${RESULTS_DIR}/fixed_random_conv_summary.csv"
EPOCH_CSV="${RESULTS_DIR}/fixed_random_conv_epochs.csv"
LOG_DIR="${RESULTS_DIR}/logs"

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --gpus)     IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MAX_PARALLEL=${#GPUS[@]}

declare -a JOBS=()
for alpha in "${ALPHAS[@]}"; do
    for cr in "${CRS[@]}"; do
        for ks in "${KERNEL_SIZES[@]}"; do
            if [[ "$alpha" == "10000000" ]]; then
                alpha_tag="iid"
            else
                alpha_tag="a${alpha}"
            fi
            # deep_decompress for ks>1, baseline single-conv for ks=1
            if (( ks > 1 )); then
                deep="yes"
                name="${DATASET}_${MODE}_cr${cr}_ks${ks}_deep_nc${NUM_CLIENTS}_${alpha_tag}"
            else
                deep="no"
                name="${DATASET}_${MODE}_cr${cr}_ks${ks}_nc${NUM_CLIENTS}_${alpha_tag}"
            fi
            JOBS+=("${cr}|${alpha}|${ks}|${deep}|${name}")
        done
    done
done

TOTAL=${#JOBS[@]}
echo "═══════════════════════════════════════════════════════════════"
echo "  Fixed Random Conv Sweep (fixed_random_conv) — kernel size ablation"
echo "  CIFAR-10 | client-heads | ${NUM_CLIENTS} clients | ${EPOCHS} epochs"
echo "  CRs: ${CRS[*]} | KS: ${KERNEL_SIZES[*]} | Alphas: ${ALPHAS[*]}"
echo "  decompress_lr: ${DECOMPRESS_LR} | recon_lambda: ${RECON_LAMBDA}"
echo "  ${TOTAL} experiments | GPUs: ${GPUS[*]}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

printf "%-6s %-4s %-6s %-12s %s\n" "CR" "KS" "Deep" "Alpha" "Name"
printf "%-6s %-4s %-6s %-12s %s\n" "------" "----" "------" "------------" "----"
for job in "${JOBS[@]}"; do
    IFS='|' read -r cr alpha ks deep name <<< "$job"
    printf "%-6s %-4s %-6s %-12s %s\n" "$cr" "$ks" "$deep" "$alpha" "$name"
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would launch ${TOTAL} experiments. Exiting."
    exit 0
fi

mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_CSV")"

declare -a SLOT_PID=()
declare -a SLOT_NAME=()
COMPLETED=0
FAILED=0

for ((i=0; i<MAX_PARALLEL; i++)); do
    SLOT_PID[$i]=0
    SLOT_NAME[$i]=""
done

reap_slots() {
    for ((i=0; i<MAX_PARALLEL; i++)); do
        if (( SLOT_PID[i] != 0 )); then
            if ! kill -0 "${SLOT_PID[$i]}" 2>/dev/null; then
                wait "${SLOT_PID[$i]}" 2>/dev/null && rc=0 || rc=$?
                if (( rc == 0 )); then
                    echo "  [DONE] ${SLOT_NAME[$i]} (GPU ${GPUS[$i]})"
                    COMPLETED=$((COMPLETED + 1))
                else
                    echo "  [FAIL] ${SLOT_NAME[$i]} (GPU ${GPUS[$i]}, exit=$rc)"
                    FAILED=$((FAILED + 1))
                fi
                SLOT_PID[$i]=0
                SLOT_NAME[$i]=""
            fi
        fi
    done
}

find_free_slot() {
    FREE_SLOT=-1
    for ((i=0; i<MAX_PARALLEL; i++)); do
        if (( SLOT_PID[i] == 0 )); then
            FREE_SLOT=$i
            return 0
        fi
    done
    return 1
}

wait_for_free_slot() {
    while true; do
        reap_slots
        if find_free_slot; then return 0; fi
        sleep 10
    done
}

echo "Dispatching experiments..."
echo ""

for job in "${JOBS[@]}"; do
    IFS='|' read -r cr alpha ks deep name <<< "$job"
    wait_for_free_slot
    gpu="${GPUS[$FREE_SLOT]}"
    logfile="${LOG_DIR}/${name}.log"
    echo "  [RUN]  ${name} → GPU ${gpu}  (log: ${logfile})"

    # Build extra flags for deep decompress
    EXTRA_FLAGS=""
    if [[ "$deep" == "yes" ]]; then
        EXTRA_FLAGS="--deep-decompress --decompress-lr ${DECOMPRESS_LR} --recon-lambda ${RECON_LAMBDA}"
    fi

    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
        --mode "$MODE" \
        --compression-ratio "$cr" \
        --conv-kernel-size "$ks" \
        --num-clients "$NUM_CLIENTS" \
        --alpha "$alpha" \
        --client-heads \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --dataset "$DATASET" \
        --gpu "$gpu" \
        --seed "$SEED" \
        --no-viz \
        --summary-csv "$SUMMARY_CSV" \
        --epoch-csv "$EPOCH_CSV" \
        $EXTRA_FLAGS \
        > "$logfile" 2>&1 &

    SLOT_PID[$FREE_SLOT]=$!
    SLOT_NAME[$FREE_SLOT]="$name"
    sleep 2
done

echo ""
echo "All jobs dispatched. Waiting for remaining..."
while true; do
    reap_slots
    all_done=true
    for ((i=0; i<MAX_PARALLEL; i++)); do
        if (( SLOT_PID[i] != 0 )); then
            all_done=false
            break
        fi
    done
    if $all_done; then break; fi
    sleep 10
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Fixed Random Conv sweep complete!"
echo "  Completed: ${COMPLETED} | Failed: ${FAILED}"
echo "  Summary: ${SUMMARY_CSV}"
echo "  Epochs:  ${EPOCH_CSV}"
echo "═══════════════════════════════════════════════════════════════"
