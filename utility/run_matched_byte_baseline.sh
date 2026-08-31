#!/bin/bash
# ============================================================================
# Matched-byte sparsification baseline (topk_sparse)
# ============================================================================
# Adds the topk_sparse column to the baseline comparison at matched transmitted
# bytes. Runs ONLY topk_sparse; the LS-F / LS-L / Raw / Learned-1x1 numbers are
# reused from the existing appendix table (full_sweep_baseline_appendix.tex), so
# this is the minimum new compute. Same protocol as run_ablation.sh and
#
#   * modes      = topk_sparse  (k_s largest-|.| coords/sample, 6 B/entry budget)
#   * datasets   = cifar10 cifar100 fmnist gtsrb   (headline cifar10 first)
#   * non-IID a  = appendix values: cifar10/fmnist/gtsrb=0.1, cifar100=0.5
#   * IID a      = 1e7  (alpha -> inf column)
#   * cells      = {L1H,L2H} x {SCH,PCH}
#   * CR         = {8,16,32}  ->  k = activation_dim/CR; topk_sparse internally
#                  keeps k_s = floor(4k/6) coords so k_s*6 B ~= k*4 B (LS budget)
#
# Per-(dataset,cell) CSV: the --sweep resume key is (alpha,nc,mode,k,client_heads,
# ks) and includes NEITHER dataset NOR deep_head, so a shared CSV could collide
# (e.g. cifar10 vs cifar100 share activation_dim=4096 -> same k). One CSV per
# (dataset,cell) keeps resume correct and makes GPU/cell parallelism safe.
#
# Resume-safe: re-run to continue; runs that reached 100 epochs are skipped.
#
# Usage (GPUs 2,3 free; split by cell so cifar10's 4 cells finish in parallel):
#   GPU=2 ./run_matched_byte_baseline.sh L1H_SCH L1H_PCH
#   GPU=3 ./run_matched_byte_baseline.sh L2H_SCH L2H_PCH
#   GPU=2 ./run_matched_byte_baseline.sh            # (no args = all 4 cells)
# ============================================================================

cd "$(dirname "$0")"   # run from the utility harness/

GPU="${GPU:-0}"
EPOCHS=100             # cosine T_max=epochs; do NOT use 2000
NUM_CLIENTS=10
CRS="8 16 32"
MODES="topk_sparse"
OUTDIR="results/csv"
ALPHA_IID="10000000"
DATASETS="${DATASETS:-cifar10 cifar100 fmnist gtsrb}"   # headline cifar10 first; override via env (e.g. DATASETS="fmnist gtsrb")

mkdir -p "$OUTDIR"

noniid_alpha() {
    case "$1" in
        cifar100)              echo "0.5" ;;
        cifar10|fmnist|gtsrb)  echo "0.1" ;;
        *)                     echo "0.1" ;;
    esac
}

# L1H = shallow (default), L2H = --deep-head ; SCH = shared (default), PCH = --client-heads
flags_for_cell() {
    case "$1" in
        L1H_SCH) echo "" ;;
        L1H_PCH) echo "--client-heads" ;;
        L2H_SCH) echo "--deep-head" ;;
        L2H_PCH) echo "--deep-head --client-heads" ;;
        *)       echo "__INVALID__" ;;
    esac
}

ALL_CELLS=(L1H_SCH L1H_PCH L2H_SCH L2H_PCH)
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=("${ALL_CELLS[@]}")

echo "========================================================================"
echo "  Matched-byte topk_sparse baseline"
echo "  GPU=${GPU}  epochs=${EPOCHS}  clients=${NUM_CLIENTS}  cells: ${CELLS[*]}"
echo "  datasets: ${DATASETS}   CR: ${CRS}"
echo "========================================================================"

FAILED=()
for ds in $DATASETS; do
    a_noniid="$(noniid_alpha "$ds")"
    for cell in "${CELLS[@]}"; do
        extra="$(flags_for_cell "$cell")"
        if [ "$extra" = "__INVALID__" ]; then
            echo "  !! unknown cell '${cell}' (valid: ${ALL_CELLS[*]}) — skipping" >&2
            continue
        fi
        sum="${OUTDIR}/mb_${ds}_${cell}_summary.csv"
        epo="${OUTDIR}/mb_${ds}_${cell}_epochs.csv"

        echo ""
        echo "------------------------------------------------------------------------"
        echo "  ${ds}  ${cell}   alphas: ${a_noniid} (non-IID), ${ALPHA_IID} (IID)"
        echo "  summary -> ${sum}"
        echo "------------------------------------------------------------------------"

        if ! python train_split.py \
            --sweep --resume \
            --dataset "$ds" \
            --epochs "$EPOCHS" \
            --num-clients "$NUM_CLIENTS" \
            --modes $MODES \
            --compression-ratio-list $CRS \
            --alpha-list "$a_noniid" "$ALPHA_IID" \
            --no-viz \
            --gpu "$GPU" \
            --summary-csv "$sum" \
            --epoch-csv "$epo" \
            $extra; then
            echo "  !! ${ds} ${cell} exited non-zero — continuing; re-run to resume" >&2
            FAILED+=("${ds}/${cell}")
        fi
    done
done

echo ""
echo "========================================================================"
echo "  Matched-byte topk_sparse sweep DONE for cells: ${CELLS[*]}"
echo "  CSVs: ${OUTDIR}/mb_<dataset>_<cell>_summary.csv"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  needs a resume re-run: ${FAILED[*]}"
fi
echo "========================================================================"
