#!/bin/bash
# Ablation study: all modes x compression ratios x features
# CIFAR-100, 10 clients, Non-IID (alpha=0.1), 200 epochs
#
# Resume-aware: skips completed runs, restarts partial runs.

set -e

GPU="${GPU:-0}"
COMMON="--dataset cifar100 --epochs 200 --num-clients 10 --alpha 0.1 --no-viz --gpu $GPU"
SUMMARY_CSV="results/csv/ablation_cifar100_noniid_summary.csv"
EPOCH_CSV="results/csv/ablation_cifar100_noniid_epochs.csv"
TARGET_EPOCHS=200

count_epochs() {
    local mode="$1" k="$2" deep="$3" rl="$4" ch="${5:-False}"
    [ ! -f "$EPOCH_CSV" ] && echo 0 && return
    awk -F',' -v m="$mode" -v kk="$k" -v d="$deep" -v r="$rl" -v c="$ch" \
        'NR>1 && $4==m && $5==kk && $13==d && $14==r && $15==c {n++} END {print n+0}' "$EPOCH_CSV"
}

purge_partial() {
    local mode="$1" k="$2" deep="$3" rl="$4" ch="${5:-False}"
    [ ! -f "$EPOCH_CSV" ] && return
    local tmp="${EPOCH_CSV}.tmp"
    awk -F',' -v m="$mode" -v kk="$k" -v d="$deep" -v r="$rl" -v c="$ch" \
        'NR==1 || !($4==m && $5==kk && $13==d && $14==r && $15==c)' "$EPOCH_CSV" > "$tmp"
    mv "$tmp" "$EPOCH_CSV"
    if [ -f "$SUMMARY_CSV" ]; then
        tmp="${SUMMARY_CSV}.tmp"
        awk -F',' -v m="$mode" -v kk="$k" -v d="$deep" -v r="$rl" -v c="$ch" \
            'NR==1 || !($2==m && $3==kk && $16==d && $17==r && $18==c)' "$SUMMARY_CSV" > "$tmp"
        mv "$tmp" "$SUMMARY_CSV"
    fi
}

run() {
    local tag="$1"
    local mode="$2" k="$3" deep="$4" rl="$5" ch="$6"
    shift 6

    local n
    n=$(count_epochs "$mode" "$k" "$deep" "$rl" "$ch")

    if [ "$n" -ge "$TARGET_EPOCHS" ]; then
        echo "  SKIP (done): $tag  [$n epochs]"
        return 0
    fi

    if [ "$n" -gt 0 ]; then
        echo "  PURGE partial: $tag  [$n/$TARGET_EPOCHS epochs] — restarting"
        purge_partial "$mode" "$k" "$deep" "$rl" "$ch"
    fi

    echo ""
    echo "========================================================================"
    echo "  ABLATION [CIFAR-100 Non-IID]: $tag"
    echo "  CMD: python train_split.py $COMMON $@ --summary-csv $SUMMARY_CSV --epoch-csv $EPOCH_CSV"
    echo "========================================================================"
    python train_split.py $COMMON "$@" --summary-csv "$SUMMARY_CSV" --epoch-csv "$EPOCH_CSV"
}

# ── Raw baseline (no compression) ──────────────────────────────────────────
run "raw"                         raw 0 False 0.0 False  --mode raw
run "raw + deep-head"             raw 0 True  0.0 False  --mode raw --deep-head

# ── Fixed Random ────────────────────────────────────────────────────────────
for CR in 8 16 32; do
    K=$((4096 / CR))
    run "fixed_random CR=${CR}"                            fixed_random $K False 0.0 False  --mode fixed_random --compression-ratio $CR
    run "fixed_random CR=${CR} + deep-head"                fixed_random $K True  0.0 False  --mode fixed_random --compression-ratio $CR --deep-head
    run "fixed_random CR=${CR} + recon-lambda"             fixed_random $K False 0.1 False  --mode fixed_random --compression-ratio $CR --recon-lambda 0.1
    run "fixed_random CR=${CR} + deep-head + recon-lambda" fixed_random $K True  0.1 False  --mode fixed_random --compression-ratio $CR --deep-head --recon-lambda 0.1
done

for CR in 8 16 32; do
    K=$((4096 / CR))
done

# ── Learned 1x1 ────────────────────────────────────────────────────────────
for CR in 8 16 32; do
    K=$((4096 / CR))
    run "learned_1x1 CR=${CR}"                            learned_1x1 $K False 0.0 False  --mode learned_1x1 --compression-ratio $CR
    run "learned_1x1 CR=${CR} + deep-head"                learned_1x1 $K True  0.0 False  --mode learned_1x1 --compression-ratio $CR --deep-head
    run "learned_1x1 CR=${CR} + recon-lambda"             learned_1x1 $K False 0.1 False  --mode learned_1x1 --compression-ratio $CR --recon-lambda 0.1
    run "learned_1x1 CR=${CR} + deep-head + recon-lambda" learned_1x1 $K True  0.1 False  --mode learned_1x1 --compression-ratio $CR --deep-head --recon-lambda 0.1
done

# ══════════════════════════════════════════════════════════════════════════════
# CLIENT-SPECIFIC HEADS (--client-heads)
# ══════════════════════════════════════════════════════════════════════════════

# ── Raw + client-heads ────────────────────────────────────────────────────────
run "raw + client-heads"                         raw 0 False 0.0 True  --mode raw --client-heads
run "raw + deep-head + client-heads"             raw 0 True  0.0 True  --mode raw --deep-head --client-heads

# ── Fixed Random + client-heads ───────────────────────────────────────────────
for CR in 8 16 32; do
    K=$((4096 / CR))
    run "fixed_random CR=${CR} + client-heads"                            fixed_random $K False 0.0 True  --mode fixed_random --compression-ratio $CR --client-heads
    run "fixed_random CR=${CR} + deep-head + client-heads"                fixed_random $K True  0.0 True  --mode fixed_random --compression-ratio $CR --deep-head --client-heads
    run "fixed_random CR=${CR} + recon-lambda + client-heads"             fixed_random $K False 0.1 True  --mode fixed_random --compression-ratio $CR --recon-lambda 0.1 --client-heads
    run "fixed_random CR=${CR} + deep-head + recon-lambda + client-heads" fixed_random $K True  0.1 True  --mode fixed_random --compression-ratio $CR --deep-head --recon-lambda 0.1 --client-heads
done

for CR in 8 16 32; do
    K=$((4096 / CR))
done

# ── Learned 1x1 + client-heads ───────────────────────────────────────────────
for CR in 8 16 32; do
    K=$((4096 / CR))
    run "learned_1x1 CR=${CR} + client-heads"                            learned_1x1 $K False 0.0 True  --mode learned_1x1 --compression-ratio $CR --client-heads
    run "learned_1x1 CR=${CR} + deep-head + client-heads"                learned_1x1 $K True  0.0 True  --mode learned_1x1 --compression-ratio $CR --deep-head --client-heads
    run "learned_1x1 CR=${CR} + recon-lambda + client-heads"             learned_1x1 $K False 0.1 True  --mode learned_1x1 --compression-ratio $CR --recon-lambda 0.1 --client-heads
    run "learned_1x1 CR=${CR} + deep-head + recon-lambda + client-heads" learned_1x1 $K True  0.1 True  --mode learned_1x1 --compression-ratio $CR --deep-head --recon-lambda 0.1 --client-heads
done

echo ""
echo "========================================================================"
echo "  ABLATION COMPLETE [CIFAR-100 Non-IID]"
echo "  Summary: $SUMMARY_CSV"
echo "  Epochs:  $EPOCH_CSV"
echo "========================================================================"
