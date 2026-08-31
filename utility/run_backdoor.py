#!/usr/bin/env python3
"""BadNets backdoor experiment under a true split-learning protocol.

True split-learning protocol: client (head + learned compress, if any) and server
(learned decompress, if any, + backbone + tail) train with separate optimizers.
Only the smashed payload z crosses the cut forward; only grad_z crosses backward.
This intentionally diverges from train_split.run_multiclient_epoch (which uses a
single monolithic optimizer); we keep SplitModel and Dirichlet partitioning from
train_split, but write our own per-batch forward/backward to enforce the cut.

Per-client per-epoch activation stats are written to detection_stats.csv. Final
summary row is written to summary.csv (extended in milestone M6).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from backdoor_data import get_backdoor_loaders
from backdoor_observer import EpochObserver
from backdoor_split import SplitForward, build_optimizers
from train_split import (
    MODEL_CONFIGS,
    SplitModel,
    accuracy_from_logits,
    set_seed,
)


# ---- ASR -------------------------------------------------------------------

def eval_asr(model: SplitModel, asr_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in asr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _ = model(x, client_id=0)
            correct += accuracy_from_logits(logits, y)
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def eval_clean(model: SplitModel, test_loaders, device: torch.device) -> float:
    """Aggregate clean test accuracy across all per-client test loaders."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cid, loader in enumerate(test_loaders):
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits, _ = model(x, client_id=cid)
                correct += accuracy_from_logits(logits, y)
                total += y.size(0)
    return correct / total if total > 0 else 0.0


# ---- CSV layout ------------------------------------------------------------

META_COLS = [
    "dataset", "mode", "k", "num_clients", "alpha",
    "malicious_ratio", "poison_rate", "target_class", "seed",
    "epoch", "client_id", "is_malicious",
]
METRIC_SUFFIXES = (
    "mean", "var", "skewness", "kurtosis",
    "l1_norm", "l2_norm", "linf_norm",
    "top_sv", "spectral_energy_ratio", "effective_rank",
    "cosine_to_global",
)
Z_COLS = [f"z_{s}" for s in METRIC_SUFFIXES]
R_COLS = [f"r_{s}" for s in METRIC_SUFFIXES]
TASK_COLS = ["loss", "accuracy", "clean_acc", "asr"]

DETECTION_COLUMNS = META_COLS + Z_COLS + R_COLS + TASK_COLS


def write_detection_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTION_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            # Ensure every column is present (pad with 0.0 if a metric was missing)
            full = {col: row.get(col, 0.0) for col in DETECTION_COLUMNS}
            writer.writerow(full)


def cohens_d(mal_vals: np.ndarray, clean_vals: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation. Returns 0.0 if undefined."""
    n1, n2 = len(mal_vals), len(clean_vals)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = float(np.mean(mal_vals)), float(np.mean(clean_vals))
    s1, s2 = float(np.std(mal_vals, ddof=1)), float(np.std(clean_vals, ddof=1))
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0
    return (m1 - m2) / pooled


def write_summary(
    detection_csv: Path,
    summary_csv: Path,
    run_meta: Dict[str, Any],
    epochs: int,
    final_ca: float,
    final_asr: float,
    best_ca: float,
    best_asr: float,
    malicious_ids: List[int],
) -> None:
    """One row per run. Cohen's d computed over the last 25% of epochs."""
    df = pd.read_csv(detection_csv)
    last_window = max(1, math.ceil(0.25 * epochs))
    cutoff = epochs - last_window + 1
    window_df = df[df["epoch"] >= cutoff]

    metric_cols = [c for c in df.columns if c.startswith("z_") or c.startswith("r_")]

    row: Dict[str, Any] = dict(run_meta)
    row["epochs"] = epochs
    row["final_ca"] = final_ca
    row["final_asr"] = final_asr
    row["best_ca"] = best_ca
    row["best_asr"] = best_asr
    row["num_malicious"] = len(malicious_ids)
    row["malicious_ids"] = ";".join(str(i) for i in malicious_ids)

    if not window_df.empty:
        mal_mask = window_df["is_malicious"] == 1
        for col in metric_cols:
            row[f"cohens_d_{col}"] = cohens_d(
                window_df.loc[mal_mask, col].to_numpy(),
                window_df.loc[~mal_mask, col].to_numpy(),
            )
    else:
        for col in metric_cols:
            row[f"cohens_d_{col}"] = 0.0

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="BadNets backdoor under true split-learning")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100"])
    # fixed_random_conv is an exploratory mode, not reported in the paper.
    p.add_argument("--mode", default="raw",
                   choices=["raw", "fixed_random",
                            "fixed_random_ll", "fixed_random_conv", "learned_1x1"])
    p.add_argument("--k", type=int, default=16,
                   help="Compression ratio (CR). k_dim = activation_dim // k. "
                        "Ignored for mode=raw.")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=10000000.0,
                   help="Dirichlet concentration. Higher = more IID. "
                        "1e7 is the IID baseline used in an earlier internal codebase.")
    p.add_argument("--malicious-ratio", type=float, default=0.1)
    p.add_argument("--poison-rate", type=float, default=0.3)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--trigger-pattern", default="square",
                   choices=["square", "cross", "l_shape"])
    p.add_argument("--trigger-size", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out-dir", default="results/backdoor")
    return p.parse_args()


# ---- Main ------------------------------------------------------------------

def _fmt_alpha(alpha: float) -> str:
    """Format alpha for use in directory names (drop trailing .0 for integers)."""
    return str(int(alpha)) if float(alpha).is_integer() else str(alpha)


def run_dir_name(args, k_dim: int) -> str:
    return (
        f"{args.dataset}_{args.mode}_c{args.num_clients}"
        f"_mal{args.malicious_ratio}_pr{args.poison_rate}"
        f"_k{k_dim}_seed{args.seed}_a{_fmt_alpha(args.alpha)}"
    )


def train_one_epoch(
    sf: SplitForward,
    train_loaders,
    obs: EpochObserver,
    device: torch.device,
) -> None:
    """Per-client per-batch loop with the manual cut at z. ~25 lines mirroring
    train_split.run_multiclient_epoch:673-706 but with split-learning semantics.
    """
    sf.model.train()
    for client_id, loader in enumerate(train_loaders):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sf.zero_grad()
            z, info, ctx = sf.client_forward(x, client_id=client_id)
            loss, logits, r, z_server = sf.server_forward(z, info, ctx, y)
            sf.backward_and_step(z, z_server, loss)

            with torch.no_grad():
                n_correct = accuracy_from_logits(logits, y)
            obs.observe(
                client_id=client_id,
                z=z.detach(),
                r=r.detach(),
                loss=float(loss.item()),
                n_correct=int(n_correct),
                bsz=int(x.size(0)),
            )


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation_dim = MODEL_CONFIGS[args.dataset]["activation_dim"]
    k_dim = 0 if args.mode == "raw" else activation_dim // args.k

    out_dir = Path(args.out_dir) / run_dir_name(args, k_dim)
    out_dir.mkdir(parents=True, exist_ok=True)
    detection_csv = out_dir / "detection_stats.csv"

    print("=" * 88)
    print(f"dataset={args.dataset}  mode={args.mode}  CR={args.k}  k_dim={k_dim}")
    print(f"clients={args.num_clients}  alpha={args.alpha}  "
          f"mal_ratio={args.malicious_ratio}  poison_rate={args.poison_rate}")
    print(f"epochs={args.epochs}  seed={args.seed}  device={device}")
    print(f"out_dir={out_dir}")
    print("=" * 88)

    # Loaders
    train_loaders, test_loaders, asr_loader, malicious_ids = get_backdoor_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        dataset=args.dataset,
        malicious_ratio=args.malicious_ratio,
        poison_rate=args.poison_rate,
        target_label=args.target_class,
        trigger_pattern=args.trigger_pattern,
        trigger_size=args.trigger_size,
    )
    print(f"malicious_ids: {malicious_ids}")

    # Model + dual optimizers
    model = SplitModel(
        dataset=args.dataset, mode=args.mode, k=k_dim,
        num_clients=args.num_clients,
    ).to(device)
    opt_client, opt_server = build_optimizers(
        model, lr=args.lr, weight_decay=args.weight_decay)
    sf = SplitForward(model, opt_client, opt_server)

    obs = EpochObserver(num_clients=args.num_clients, malicious_ids=malicious_ids)

    run_meta = {
        "dataset": args.dataset,
        "mode": args.mode,
        "k": k_dim,
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "malicious_ratio": args.malicious_ratio,
        "poison_rate": args.poison_rate,
        "target_class": args.target_class,
        "seed": args.seed,
    }

    best_ca = 0.0
    best_asr = 0.0
    final_ca = 0.0
    final_asr = 0.0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(sf, train_loaders, obs, device)

        ca = eval_clean(model, test_loaders, device)
        asr = eval_asr(model, asr_loader, device)
        best_ca = max(best_ca, ca)
        best_asr = max(best_asr, asr)
        final_ca, final_asr = ca, asr

        rows = obs.finalize_epoch(epoch)
        for r in rows:
            r.update(run_meta)
            r["clean_acc"] = ca
            r["asr"] = asr
        write_detection_rows(detection_csv, rows)

        print(f"epoch {epoch:03d} | CA {ca*100:6.2f}% | ASR {asr*100:6.2f}%")

    summary_csv = out_dir / "summary.csv"
    write_summary(
        detection_csv=detection_csv,
        summary_csv=summary_csv,
        run_meta=run_meta,
        epochs=args.epochs,
        final_ca=final_ca,
        final_asr=final_asr,
        best_ca=best_ca,
        best_asr=best_asr,
        malicious_ids=malicious_ids,
    )

    print(f"\nWrote {detection_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
