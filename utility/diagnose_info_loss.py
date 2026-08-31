#!/usr/bin/env python3
"""
Diagnose information loss through the ks=9 random conv bottleneck.

Three diagnostics:
  1. SVD of unrolled conv operator — condition number, energy preservation, effective rank
  2. Linear probe — train a linear classifier on compressed features directly
     (if this gets ~82%, the info is there and the problem is backbone adaptation)
  3. Nonlinear probe — small MLP on compressed features
     (upper bound on what's extractable from compressed representation)

Compare ks=1 vs ks=9 to isolate the spatial mixing effect.
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from client_models import build_head, MODEL_CONFIGS


def build_compress(in_channels: int, k_channels: int, kernel_size: int) -> nn.Conv2d:
    """Build a frozen random orthogonal conv (same as FixedRandomConvBottleneck)."""
    pad = kernel_size // 2
    conv = nn.Conv2d(in_channels, k_channels, kernel_size, padding=pad, bias=False)
    with torch.no_grad():
        w = conv.weight.data
        flat = w.view(w.shape[0], -1)
        q, _ = torch.linalg.qr(flat.T)
        conv.weight.data = q.T.view_as(w)
    conv.weight.requires_grad_(False)
    return conv


# ── Diagnostic 1: SVD analysis ──────────────────────────────────────────────
def analyze_svd(in_ch: int, out_ch: int, kernel_size: int, spatial: int):
    """Unroll Conv2d into full matrix, compute SVD."""
    print(f"\n{'='*60}")
    print(f"  SVD Analysis: Conv2d({in_ch}→{out_ch}, ks={kernel_size}) on {spatial}×{spatial}")
    print(f"{'='*60}")

    compress = build_compress(in_ch, out_ch, kernel_size)

    d_in = in_ch * spatial * spatial
    d_out = out_ch * spatial * spatial

    # Build full matrix by passing basis vectors
    W = torch.zeros(d_out, d_in)
    with torch.no_grad():
        for i in range(d_in):
            e = torch.zeros(1, in_ch, spatial, spatial)
            e.view(-1)[i] = 1.0
            out = compress(e)
            W[:, i] = out.view(-1)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    rank = (S > 1e-6).sum().item()

    print(f"  Matrix shape     : {W.shape[0]} × {W.shape[1]}")
    print(f"  Rank             : {rank} / {min(d_out, d_in)}")
    print(f"  σ_max            : {S[0]:.4f}")
    print(f"  σ_min (nonzero)  : {S[S > 1e-6][-1]:.4f}")
    print(f"  Condition number : {S[0] / S[S > 1e-6][-1]:.4f}")
    print(f"  Null space dim   : {d_in - rank}")

    # Energy preservation: how much of random input energy survives
    x = torch.randn(500, d_in)
    y = x @ W.T
    x_recon = y @ W  # W^T W x (project onto range of W^T)
    energy_in = (x ** 2).sum(dim=1).mean()
    energy_preserved = ((x_recon) ** 2).sum(dim=1).mean() / energy_in
    recon_mse = ((x - x_recon) ** 2).sum(dim=1).mean() / energy_in
    print(f"  Energy preserved : {energy_preserved:.4f} ({energy_preserved*100:.1f}%)")
    print(f"  Relative MSE     : {recon_mse:.4f}")

    # Singular value spectrum
    print(f"  Top 10 σ         : {S[:10].tolist()}")
    if rank > 10:
        print(f"  Bottom 10 σ      : {S[rank-10:rank].tolist()}")

    # Effective rank (Shannon entropy of normalized singular values)
    s_norm = S[S > 1e-6] / S[S > 1e-6].sum()
    eff_rank = torch.exp(-torch.sum(s_norm * torch.log(s_norm))).item()
    print(f"  Effective rank   : {eff_rank:.1f} / {rank}")

    return W, S


# ── Diagnostic 2 & 3: Probe classifiers ─────────────────────────────────────
@torch.no_grad()
def extract_features(head, compress, dataloader, device):
    """Extract compressed features and labels from dataset."""
    all_feats, all_labels = [], []
    head.eval()
    for x, y in dataloader:
        x = x.to(device)
        z = head(x)
        c = compress(z)
        all_feats.append(c.flatten(1).cpu())
        all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_probe(feats_train, labels_train, feats_test, labels_test,
                probe, device, epochs=50, lr=1e-3):
    """Train a probe classifier and return best test accuracy."""
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    # Mini-batch training
    n = feats_train.shape[0]
    bs = 256
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n)
        total_loss, total_correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            x = feats_train[idx].to(device)
            y = labels_train[idx].to(device)
            logits = probe(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(1) == y).sum().item()

        train_acc = total_correct / n

        # Eval
        probe.eval()
        with torch.no_grad():
            test_logits = probe(feats_test.to(device))
            test_acc = (test_logits.argmax(1) == labels_test.to(device)).float().mean().item()
        best_acc = max(best_acc, test_acc)

        if ep % 10 == 0 or ep == epochs:
            print(f"    epoch {ep:3d} | train_acc {train_acc*100:5.1f}% | test_acc {test_acc*100:5.1f}% | best {best_acc*100:5.1f}%")

    return best_acc


def run_probes(kernel_size: int, device: torch.device, seed: int = 42):
    """Run linear and MLP probes on compressed features."""
    print(f"\n{'='*60}")
    print(f"  Probe Classifiers: ks={kernel_size}, CR=8")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("./data", train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    # Head + compress
    head = build_head("cifar10", deep=False).to(device)
    head.eval()
    in_ch = 64
    k_ch = 8  # CR=8
    compress = build_compress(in_ch, k_ch, kernel_size).to(device)

    feat_dim = k_ch * 8 * 8  # 512

    print(f"  Extracting features (feat_dim={feat_dim})...")
    feats_train, labels_train = extract_features(head, compress, train_loader, device)
    feats_test, labels_test = extract_features(head, compress, test_loader, device)
    print(f"  Train: {feats_train.shape}, Test: {feats_test.shape}")

    # Also extract raw head features for comparison
    raw_compress = build_compress(in_ch, in_ch, 1).to(device)  # identity-like
    raw_compress.weight.data = torch.eye(in_ch, device=device).view(in_ch, in_ch, 1, 1)
    raw_train, _ = extract_features(head, raw_compress, train_loader, device)
    raw_test, _ = extract_features(head, raw_compress, test_loader, device)

    # Feature statistics
    print(f"\n  Feature statistics:")
    print(f"    Compressed — mean: {feats_train.mean():.4f}, std: {feats_train.std():.4f}, "
          f"min: {feats_train.min():.4f}, max: {feats_train.max():.4f}")
    print(f"    Raw head   — mean: {raw_train.mean():.4f}, std: {raw_train.std():.4f}, "
          f"min: {raw_train.min():.4f}, max: {raw_train.max():.4f}")

    # Effective dimensionality (PCA)
    feats_centered = feats_train - feats_train.mean(0)
    _, sv, _ = torch.linalg.svd(feats_centered, full_matrices=False)
    sv_norm = sv / sv.sum()
    eff_dim = torch.exp(-torch.sum(sv_norm * torch.log(sv_norm + 1e-10))).item()
    cumvar = torch.cumsum(sv ** 2, dim=0) / (sv ** 2).sum()
    dims_95 = (cumvar < 0.95).sum().item() + 1
    dims_99 = (cumvar < 0.99).sum().item() + 1
    print(f"    Effective dim     : {eff_dim:.1f} / {feat_dim}")
    print(f"    Dims for 95% var  : {dims_95}")
    print(f"    Dims for 99% var  : {dims_99}")

    # Linear probe
    print(f"\n  Linear probe (LogisticRegression on {feat_dim}-d compressed features):")
    linear_probe = nn.Linear(feat_dim, 10)
    linear_acc = train_probe(feats_train, labels_train, feats_test, labels_test,
                             linear_probe, device, epochs=50)

    # MLP probe
    print(f"\n  MLP probe (2-layer, 256-hidden):")
    mlp_probe = nn.Sequential(
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    mlp_acc = train_probe(feats_train, labels_train, feats_test, labels_test,
                          mlp_probe, device, epochs=100)

    # Raw features probe (upper bound)
    print(f"\n  Linear probe on RAW head features ({raw_train.shape[1]}-d, upper bound):")
    raw_probe = nn.Linear(raw_train.shape[1], 10)
    raw_acc = train_probe(raw_train, labels_train, raw_test, labels_test,
                          raw_probe, device, epochs=50)

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  SUMMARY: ks={kernel_size}, CR=8               │")
    print(f"  │  Linear probe (compressed) : {linear_acc*100:5.1f}%       │")
    print(f"  │  MLP probe (compressed)    : {mlp_acc*100:5.1f}%       │")
    print(f"  │  Linear probe (raw head)   : {raw_acc*100:5.1f}%       │")
    print(f"  │  Split learning (actual)   : 70.95%       │")
    print(f"  │  Raw split learning        : 86.86%       │")
    print(f"  └──────────────────────────────────────────┘")

    return linear_acc, mlp_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11],
                        help="Kernel sizes to sweep")
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    kernel_sizes = args.kernel_sizes

    # SVD analysis for all kernel sizes
    for ks in kernel_sizes:
        analyze_svd(in_ch=64, out_ch=8, kernel_size=ks, spatial=8)

    # Probe classifiers for all kernel sizes
    results = {}
    for ks in kernel_sizes:
        print(f"\n{'#'*60}")
        print(f"  KERNEL SIZE = {ks}")
        print(f"{'#'*60}")
        linear_acc, mlp_acc = run_probes(kernel_size=ks, device=device, seed=args.seed)
        results[ks] = (linear_acc, mlp_acc)

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — Probe accuracy vs kernel size (CR=8)")
    print(f"{'='*70}")
    print(f"  {'KS':>4s}  {'Linear':>8s}  {'MLP':>8s}  {'Split (actual)':>14s}  {'Gap (MLP-Split)':>15s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'--------------':>14s}  {'---------------':>15s}")

    # Known split learning results
    split_results = {1: 82.22, 3: 78.59, 5: 73.49, 9: 70.95}

    for ks in kernel_sizes:
        lin, mlp = results[ks]
        split = split_results.get(ks, None)
        split_str = f"{split:.2f}%" if split else "   —"
        gap_str = f"{(mlp*100 - split):.2f}%" if split else "   —"
        print(f"  {ks:4d}  {lin*100:7.2f}%  {mlp*100:7.2f}%  {split_str:>14s}  {gap_str:>15s}")

    print(f"\n  If MLP >> Split: information is there, backbone can't use it")
    print(f"  If MLP ≈ Split:  information is genuinely lost by compression")


if __name__ == "__main__":
    main()
