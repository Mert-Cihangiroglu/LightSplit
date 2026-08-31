#!/usr/bin/env python3
"""Privileged attacker security analysis for fixed random orthogonal projection.

Five experiments that evaluate the defense under a maximally generous threat
model where the attacker *knows* the projection matrix R:

  Exp 1 — Pseudoinverse recovery baseline (linear, no training)
  Exp 2 — Oracle nonlinear decoder with known R (powerful MLP, residual)
  Exp 3 — Known-R vs unknown-R image reconstruction comparison
  Exp 4 — Null-space energy analysis (pure computation)
  Exp 5 — Per-class reconstruction quality breakdown

Usage:
    python privileged_attack.py --cr-list 8 16 32 --experiments 1 2 3 4 5
    python privileged_attack.py --experiments 4          # quick smoke test
    python privileged_attack.py --load-split-checkpoint results/privileged_attack/checkpoints
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train_split import (
    SplitModel, FixedRandomBottleneck, FixedRandomMLPLiftBottleneck,
    set_seed, DATASET_STATS, accuracy_from_logits, count_parameters,
    get_datasets, append_csv_row,
)
from client_models import MODEL_CONFIGS
from sdar_attack import denormalize, compute_ssim

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_C, HEAD_H, HEAD_W = 64, 8, 8


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
_LPIPS_FN = None


def _get_lpips(device: torch.device):
    """Lazy-load LPIPS (alex) with graceful fallback."""
    global _LPIPS_FN
    if _LPIPS_FN is not None:
        return _LPIPS_FN
    try:
        import lpips
        _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False).to(device)
        _LPIPS_FN.eval()
    except ImportError:
        try:
            from piq import LPIPS as PiqLPIPS
            _LPIPS_FN = PiqLPIPS().to(device)
            _LPIPS_FN.eval()
        except ImportError:
            print("WARNING: neither lpips nor piq installed — LPIPS will be NaN")
            _LPIPS_FN = None
    return _LPIPS_FN


def compute_psnr_batch(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean PSNR over a batch of [0,1] images. Per-sample then averaged."""
    mse_per = ((img1 - img2) ** 2).flatten(1).mean(1)
    # Clamp mse to avoid inf
    mse_per = mse_per.clamp(min=1e-10)
    psnr_per = 10.0 * torch.log10(1.0 / mse_per)
    return psnr_per.mean().item()


def compute_lpips_batch(img1: torch.Tensor, img2: torch.Tensor,
                        device: torch.device) -> float:
    """Mean LPIPS over a batch of [0,1] images."""
    fn = _get_lpips(device)
    if fn is None:
        return float("nan")
    # Co-locate model and inputs: the cached LPIPS module (and its registered
    # buffers, e.g. the ScalingLayer shift/scale) may live on a different device
    # than the images, which raises "tensors on two devices" inside LPIPS.
    fn = fn.to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    with torch.no_grad():
        # LPIPS expects [-1, 1]
        val = fn(img1 * 2 - 1, img2 * 2 - 1)
    return val.mean().item()


@dataclass
class ReconMetrics:
    mse: float
    ssim: float
    psnr: float
    lpips: float


def compute_recon_metrics(recon: torch.Tensor, orig: torch.Tensor,
                          device: torch.device) -> ReconMetrics:
    """Compute all reconstruction metrics on [0,1] image batches."""
    return ReconMetrics(
        mse=F.mse_loss(recon, orig).item(),
        ssim=compute_ssim(recon, orig),
        psnr=compute_psnr_batch(recon, orig),
        lpips=compute_lpips_batch(recon, orig, device),
    )


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_activations(
    model: SplitModel,
    loader: DataLoader,
    device: torch.device,
    dataset: str,
    max_samples: int = 50000,
) -> Dict[str, torch.Tensor]:
    """Collect (h, z, x01, y) from a trained split model.

    h   — raw head activations, flat (N, d)
    z   — compressed activations (N, k)
    x01 — denormalized images [0,1] (N, 3, 32, 32)
    y   — labels (N,)
    """
    model.eval()

    spatial_bottleneck = model.mode in ("learned_1x1", "fixed_random_conv")

    hs, zs, xs, ys = [], [], [], []
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            z_spatial = model.head(x)
            h = z_spatial.flatten(1)  # (B, d)
            if spatial_bottleneck:
                compressed, _ = model.bottleneck(z_spatial)
                z = compressed.flatten(1)  # (B, k_ch*H*W)
            else:
                z, _ = model.bottleneck(h)  # (B, k)
        x01 = denormalize(x, dataset)

        hs.append(h.cpu())
        zs.append(z.cpu())
        xs.append(x01.cpu())
        ys.append(y)

        total += x.size(0)
        if total >= max_samples:
            break

    return {
        "h": torch.cat(hs)[:max_samples],
        "z": torch.cat(zs)[:max_samples],
        "x01": torch.cat(xs)[:max_samples],
        "y": torch.cat(ys)[:max_samples],
    }


def collect_activations_grads(
    model: SplitModel,
    loader: DataLoader,
    device: torch.device,
    dataset: str,
    max_samples: int = 50000,
) -> Dict[str, torch.Tensor]:
    """Collect (z, g, x01, y) where g = dL/dz is the returned cut-layer gradient.

    Unlike collect_activations (which runs under no_grad), this runs WITH autograd:
    for each batch it does a forward pass, reads the transmitted z =
    info["bottleneck_z"], forms L = CE(logits, y), and back-propagates to z.
    g = dL/dz is exactly the gradient the honest-but-curious server returns across
    the cut. The model is fixed (eval mode); only z's gradient is taken. The
    attacker computes g on its own auxiliary labels here; at deployment it observes
    the same gradient flowing back across the cut.
    """
    model.eval()
    zs, gs, xs, ys = [], [], [], []
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, info = model(x)
        if "bottleneck_z" not in info:
            raise RuntimeError(
                "collect_activations_grads needs a mode that exposes "
                "info['bottleneck_z'] (e.g. fixed_random); got mode without it.")
        z = info["bottleneck_z"]
        loss = F.cross_entropy(logits, y, reduction="sum")
        g = torch.autograd.grad(loss, z)[0]   # (B, k) per-sample dL/dz
        x01 = denormalize(x, dataset)

        zs.append(z.detach().cpu())
        gs.append(g.detach().cpu())
        xs.append(x01.detach().cpu())
        ys.append(y.detach().cpu())

        total += x.size(0)
        if total >= max_samples:
            break

    return {
        "z": torch.cat(zs)[:max_samples],
        "g": torch.cat(gs)[:max_samples],
        "x01": torch.cat(xs)[:max_samples],
        "y": torch.cat(ys)[:max_samples],
    }


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class ActivationDecoder(nn.Module):
    """Decode full-dim activations (d=4096) back to 32x32 RGB images.

    Architecture mirrors the flat-input Decoder from sdar_attack.py.
    Trained on raw (h, x01) pairs.
    """

    def __init__(self, activation_dim: int = 4096):
        super().__init__()
        self.fc = nn.Linear(activation_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),                   # 4 -> 8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 8 -> 16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 16 -> 32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(h).view(-1, 256, 4, 4))


class OracleActivationAttacker(nn.Module):
    """Attacker who trains an optimal nonlinear decoder.

    When R is known, recovers full-dim activations via residual learning:
        h_hat = pseudoinverse(z) + MLP(z)
    When R is None (e.g. learned_1x1), uses pure MLP:
        h_hat = MLP(z)
    """

    def __init__(self, R: Optional[torch.Tensor], d: int, k: int):
        super().__init__()
        if R is not None:
            self.register_buffer("R", R)
        else:
            self.R = None
        self.decoder = nn.Sequential(
            nn.Linear(k, 4 * d),
            nn.BatchNorm1d(4 * d),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d, 4 * d),
            nn.BatchNorm1d(4 * d),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d, d),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h_refined = self.decoder(z)
        if self.R is not None:
            h_pseudo = z @ self.R.T
            return h_pseudo + h_refined  # residual learning
        return h_refined


class KnownRImageDecoder(nn.Module):
    """Direct z -> image decoder with known R.

    Input: cat(z, z @ R.T) — attacker gets both compressed and pseudoinverse.
    """

    def __init__(self, R: torch.Tensor, d: int, k: int):
        super().__init__()
        self.register_buffer("R", R)
        in_dim = k + d
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),                   # 4 -> 8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 8 -> 16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 16 -> 32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h_pseudo = z @ self.R.T
        combined = torch.cat([z, h_pseudo], dim=1)
        return self.net(self.fc(combined).view(-1, 256, 4, 4))


class UnknownRImageDecoder(nn.Module):
    """Direct z -> image decoder without R knowledge.

    To keep parameter count comparable to KnownRImageDecoder, we use a wider
    hidden layer in the FC projection.
    """

    def __init__(self, k: int, d: int):
        super().__init__()
        # Wider FC to match parameter count of KnownRImageDecoder
        # KnownR has fc: (k+d) * 256*4*4 params.  We match with two-layer FC.
        self.fc = nn.Sequential(
            nn.Linear(k, 256 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Linear(256 * 4 * 4, 256 * 4 * 4),
        )
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),                   # 4 -> 8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 8 -> 16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 16 -> 32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z).view(-1, 256, 4, 4))


class GradFusionImageDecoder(nn.Module):
    """Adaptive attacker fusing the forward activation z and the returned gradient g.

    Input is cat(z, g) — the two signals the honest-but-curious server observes
    across the cut (projected activations + returned gradients). When fuse_pinv=True
    and R is known, the pseudoinverse channel z @ R.T is appended too, giving the
    strongest "combine all signals" attacker. Mirrors KnownRImageDecoder's stack.
    """

    def __init__(self, k: int, d: int, R: Optional[torch.Tensor] = None,
                 fuse_pinv: bool = False):
        super().__init__()
        self.fuse_pinv = bool(fuse_pinv and R is not None)
        if self.fuse_pinv:
            self.register_buffer("R", R)
        else:
            self.R = None
        in_dim = k + k + (d if self.fuse_pinv else 0)
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),                   # 4 -> 8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 8 -> 16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),                   # 16 -> 32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        parts = [z, g]
        if self.fuse_pinv:
            parts.append(z @ self.R.T)
        combined = torch.cat(parts, dim=1)
        return self.net(self.fc(combined).view(-1, 256, 4, 4))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train_model_mse(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 5e-4,
    label: str = "model",
) -> nn.Module:
    """Train a model with MSE loss on (input, target) pairs."""
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_x.size(0)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        count = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = train_x[idx].to(device, non_blocking=True)
            y = train_y[idx].to(device, non_blocking=True)

            pred = model(x)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(idx)
            count += len(idx)

        scheduler.step()

        if ep % 50 == 0 or ep == epochs:
            print(f"  [{label}] epoch {ep:3d}/{epochs}  "
                  f"MSE={total_loss / count:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    return model


def _train_split_model(
    model: SplitModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    patience: int = 100,
    lr: float = 1e-3,
    recon_lambda: float = 0.0,
    wcc_lambda: float = 0.0,
) -> SplitModel:
    """Train a SplitModel with fixed_random bottleneck."""
    k = model.k
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, info = model(x)
            loss = F.cross_entropy(logits, y)
            if recon_lambda > 0 and "recon_loss" in info:
                loss = loss + recon_lambda * info["recon_loss"]
            if wcc_lambda > 0 and "bottleneck_z" in info:
                z_bn = info["bottleneck_z"]
                wcc_loss = torch.tensor(0.0, device=device)
                for c in y.unique():
                    mask = (y == c)
                    if mask.sum() > 1:
                        z_c = z_bn[mask]
                        mu_c = z_c.mean(dim=0, keepdim=True)
                        wcc_loss = wcc_loss + (z_c - mu_c).pow(2).sum(dim=1).mean()
                loss = loss + wcc_lambda * wcc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += accuracy_from_logits(logits, y)
            total_samples += x.size(0)

        scheduler.step()
        train_acc = total_correct / total_samples

        # Evaluate
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits, _ = model(x)
                test_correct += accuracy_from_logits(logits, y)
                test_total += y.size(0)
        test_acc = test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if ep % 20 == 0 or ep == epochs:
            print(f"  [SplitModel k={k}] epoch {ep:3d}/{epochs}  "
                  f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"best={best_acc:.4f}")

        if patience > 0 and epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {ep} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  SplitModel k={k} training complete.  best_acc={best_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Experiment 0: Raw Baseline (no projection)
# ---------------------------------------------------------------------------
def run_exp0_raw_baseline(
    act_decoder: ActivationDecoder,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
) -> Dict[str, float]:
    """Exp 0: Raw baseline — decoder applied to uncompressed activations.

    This is the upper bound: what the ActivationDecoder can achieve with
    full information (no projection).  Shows the decoder itself is not
    the bottleneck.
    """
    print("\n" + "=" * 70)
    print("  EXP 0: Raw Baseline (no projection)")
    print("=" * 70)

    h, x01 = data["h"], data["x01"]
    act_decoder.eval()

    all_recon = []
    bs = 256
    for i in range(0, h.size(0), bs):
        with torch.no_grad():
            recon = act_decoder(h[i:i + bs].to(device))
        all_recon.append(recon.cpu())
    recon_all = torch.cat(all_recon)

    metrics = compute_recon_metrics(recon_all, x01, device)
    print(f"  Image-space:  MSE={metrics.mse:.6f}  SSIM={metrics.ssim:.4f}  "
          f"PSNR={metrics.psnr:.2f}  LPIPS={metrics.lpips:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(32, x01.size(0))
    grid = torch.cat([x01[:n_show], recon_all[:n_show]], dim=0)
    save_image(grid, out_dir / "samples_raw.png", nrow=n_show, padding=2)

    result = {
        "experiment": "exp0_raw_baseline", "k": "none", "cr": 1,
        "img_mse": metrics.mse, "img_ssim": metrics.ssim,
        "img_psnr": metrics.psnr, "img_lpips": metrics.lpips,
    }
    return result


# ---------------------------------------------------------------------------
# Experiment 1: Pseudoinverse Recovery Baseline
# ---------------------------------------------------------------------------
def run_exp1_pseudoinverse(
    R: torch.Tensor,
    act_decoder: ActivationDecoder,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    k: int,
) -> Dict[str, float]:
    """Exp 1: Pseudoinverse recovery — best-case linear attacker."""
    print("\n" + "=" * 70)
    print(f"  EXP 1: Pseudoinverse Recovery  (k={k})")
    print("=" * 70)

    h, z, x01, y = data["h"], data["z"], data["x01"], data["y"]
    R_dev = R.to(device)
    act_decoder.eval()

    # Pseudoinverse recovery
    h_hat = z.to(device) @ R_dev.T  # (N, d)

    # Activation-space error
    h_dev = h.to(device)
    rel_error = ((h_hat - h_dev) ** 2).sum(1) / (h_dev ** 2).sum(1).clamp(min=1e-10)
    act_mse = F.mse_loss(h_hat, h_dev).item()
    print(f"  Activation-space:  MSE={act_mse:.6f}  "
          f"rel_error={rel_error.mean().item():.4f} +/- {rel_error.std().item():.4f}")

    # Image reconstruction (in batches to avoid OOM)
    all_recon = []
    bs = 256
    for i in range(0, h_hat.size(0), bs):
        with torch.no_grad():
            recon = act_decoder(h_hat[i:i + bs])
        all_recon.append(recon.cpu())
    recon_all = torch.cat(all_recon)

    metrics = compute_recon_metrics(recon_all, x01, device)
    print(f"  Image-space:  MSE={metrics.mse:.6f}  SSIM={metrics.ssim:.4f}  "
          f"PSNR={metrics.psnr:.2f}  LPIPS={metrics.lpips:.4f}")

    # Save samples
    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(32, x01.size(0))
    grid = torch.cat([x01[:n_show], recon_all[:n_show]], dim=0)
    save_image(grid, out_dir / f"samples_k{k}.png", nrow=n_show, padding=2)

    result = {
        "experiment": "exp1_pseudoinverse", "k": k, "cr": 4096 // k,
        "act_mse": act_mse,
        "act_rel_error_mean": rel_error.mean().item(),
        "act_rel_error_std": rel_error.std().item(),
        "img_mse": metrics.mse, "img_ssim": metrics.ssim,
        "img_psnr": metrics.psnr, "img_lpips": metrics.lpips,
    }
    return result


# ---------------------------------------------------------------------------
# Experiment 2: Oracle Nonlinear Decoder
# ---------------------------------------------------------------------------
def run_exp2_oracle(
    R: Optional[torch.Tensor],
    act_decoder: ActivationDecoder,
    attacker_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    k: int,
    d: int,
    attack_epochs: int = 200,
) -> Dict[str, float]:
    """Exp 2: Oracle nonlinear decoder (with R residual if available, pure MLP otherwise)."""
    print("\n" + "=" * 70)
    tag = "Oracle MLP + R residual" if R is not None else "Oracle MLP (no R)"
    print(f"  EXP 2: {tag}  (k={k})")
    print("=" * 70)

    oracle = OracleActivationAttacker(R, d, k)
    print(f"  Oracle params: {count_parameters(oracle):,}")

    # Train on (z, h) pairs
    oracle = _train_model_mse(
        oracle, attacker_data["z"], attacker_data["h"],
        device, epochs=attack_epochs, lr=1e-3, label="Oracle",
    )

    # Evaluate on test set
    z_test = test_data["z"]
    h_test = test_data["h"]
    x01_test = test_data["x01"]
    act_decoder.eval()

    all_h_hat, all_recon = [], []
    bs = 256
    for i in range(0, z_test.size(0), bs):
        with torch.no_grad():
            h_hat = oracle(z_test[i:i + bs].to(device))
            recon = act_decoder(h_hat)
        all_h_hat.append(h_hat.cpu())
        all_recon.append(recon.cpu())

    h_hat_all = torch.cat(all_h_hat)
    recon_all = torch.cat(all_recon)

    # Activation-space metrics
    rel_error = ((h_hat_all - h_test) ** 2).sum(1) / (h_test ** 2).sum(1).clamp(min=1e-10)
    act_mse = F.mse_loss(h_hat_all, h_test).item()
    print(f"  Activation-space:  MSE={act_mse:.6f}  "
          f"rel_error={rel_error.mean().item():.4f} +/- {rel_error.std().item():.4f}")

    # Image-space metrics
    metrics = compute_recon_metrics(recon_all, x01_test, device)
    print(f"  Image-space:  MSE={metrics.mse:.6f}  SSIM={metrics.ssim:.4f}  "
          f"PSNR={metrics.psnr:.2f}  LPIPS={metrics.lpips:.4f}")

    # Save samples
    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(32, x01_test.size(0))
    grid = torch.cat([x01_test[:n_show], recon_all[:n_show]], dim=0)
    save_image(grid, out_dir / f"samples_k{k}.png", nrow=n_show, padding=2)

    result = {
        "experiment": "exp2_oracle", "k": k, "cr": d // k,
        "act_mse": act_mse,
        "act_rel_error_mean": rel_error.mean().item(),
        "act_rel_error_std": rel_error.std().item(),
        "img_mse": metrics.mse, "img_ssim": metrics.ssim,
        "img_psnr": metrics.psnr, "img_lpips": metrics.lpips,
    }
    return result, oracle, recon_all


# ---------------------------------------------------------------------------
# Experiment 3: Known-R vs Unknown-R
# ---------------------------------------------------------------------------
def run_exp3_known_vs_unknown(
    R: Optional[torch.Tensor],
    attacker_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    k: int,
    d: int,
    attack_epochs: int = 200,
) -> Dict[str, float]:
    """Exp 3: Known-R vs Unknown-R image reconstruction comparison.

    When R is None (learned_1x1), only Unknown-R is run.
    """
    print("\n" + "=" * 70)
    print(f"  EXP 3: {'Known-R vs ' if R is not None else ''}Unknown-R  (k={k})")
    print("=" * 70)

    known_metrics = None
    known_recon = None

    if R is not None:
        # --- Known-R decoder: input = cat(z, z @ R.T) ---
        known_dec = KnownRImageDecoder(R, d, k)
        print(f"  Known-R decoder params: {count_parameters(known_dec):,}")

        known_dec = _train_image_decoder(
            known_dec, attacker_data["z"], attacker_data["x01"],
            device, epochs=attack_epochs, label="Known-R",
        )

    # --- Unknown-R decoder: input = z ---
    unknown_dec = UnknownRImageDecoder(k, d)
    print(f"  Unknown-R decoder params: {count_parameters(unknown_dec):,}")

    unknown_dec = _train_image_decoder(
        unknown_dec, attacker_data["z"], attacker_data["x01"],
        device, epochs=attack_epochs, label="Unknown-R",
    )

    # Evaluate
    x01_test = test_data["x01"]
    z_test = test_data["z"]
    bs = 256

    known_recons, unknown_recons = [], []
    for i in range(0, z_test.size(0), bs):
        zb = z_test[i:i + bs].to(device)
        with torch.no_grad():
            if R is not None:
                known_recons.append(known_dec(zb).cpu())
            unknown_recons.append(unknown_dec(zb).cpu())

    unknown_recon = torch.cat(unknown_recons)
    unknown_metrics = compute_recon_metrics(unknown_recon, x01_test, device)

    if R is not None:
        known_recon = torch.cat(known_recons)
        known_metrics = compute_recon_metrics(known_recon, x01_test, device)
        print(f"  Known-R:   MSE={known_metrics.mse:.6f}  SSIM={known_metrics.ssim:.4f}  "
              f"PSNR={known_metrics.psnr:.2f}  LPIPS={known_metrics.lpips:.4f}")

    print(f"  Unknown-R: MSE={unknown_metrics.mse:.6f}  SSIM={unknown_metrics.ssim:.4f}  "
          f"PSNR={unknown_metrics.psnr:.2f}  LPIPS={unknown_metrics.lpips:.4f}")

    # Save comparison
    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(16, x01_test.size(0))
    grid_parts = [x01_test[:n_show]]
    if known_recon is not None:
        grid_parts.append(known_recon[:n_show])
    grid_parts.append(unknown_recon[:n_show])
    grid = torch.cat(grid_parts, dim=0)
    save_image(grid, out_dir / f"comparison_k{k}.png", nrow=n_show, padding=2)

    result = {
        "experiment": "exp3_known_vs_unknown", "k": k, "cr": d // k,
        "unknown_mse": unknown_metrics.mse, "unknown_ssim": unknown_metrics.ssim,
        "unknown_psnr": unknown_metrics.psnr, "unknown_lpips": unknown_metrics.lpips,
    }
    if known_metrics is not None:
        result.update({
            "known_mse": known_metrics.mse, "known_ssim": known_metrics.ssim,
            "known_psnr": known_metrics.psnr, "known_lpips": known_metrics.lpips,
        })
    return result


def _train_image_decoder(
    decoder: nn.Module,
    train_z: torch.Tensor,
    train_x01: torch.Tensor,
    device: torch.device,
    epochs: int = 200,
    label: str = "decoder",
    train_g: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Train an image decoder (z -> image) with MSE loss.

    If train_g is given, the decoder is called as decoder(z, g) — used by the
    adaptive activations+gradients attacker (Exp 6); otherwise decoder(z).
    """
    decoder = decoder.to(device).train()
    opt = torch.optim.Adam(decoder.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_z.size(0)
    bs = 256

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total_loss, count = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            z = train_z[idx].to(device, non_blocking=True)
            x = train_x01[idx].to(device, non_blocking=True)

            if train_g is not None:
                gb = train_g[idx].to(device, non_blocking=True)
                recon = decoder(z, gb)
            else:
                recon = decoder(z)
            loss = F.mse_loss(recon, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(idx)
            count += len(idx)

        scheduler.step()

        if ep % 50 == 0 or ep == epochs:
            print(f"  [{label}] epoch {ep:3d}/{epochs}  "
                  f"MSE={total_loss / count:.6f}")

    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Experiment 4: Null-Space Energy Analysis
# ---------------------------------------------------------------------------
def run_exp4_nullspace(
    R: torch.Tensor,
    data: Dict[str, torch.Tensor],
    k: int,
    d: int,
) -> Dict[str, float]:
    """Exp 4: Null-space energy analysis — pure computation."""
    print("\n" + "=" * 70)
    print(f"  EXP 4: Null-Space Energy Analysis  (k={k})")
    print("=" * 70)

    h = data["h"]  # (N, d)
    R_cpu = R.cpu()

    z = h @ R_cpu                    # project
    h_recovered = z @ R_cpu.T        # pseudoinverse
    residual = h - h_recovered       # null-space component

    total_energy = (h ** 2).sum(dim=1)
    recovered_energy = (h_recovered ** 2).sum(dim=1)
    lost_energy = (residual ** 2).sum(dim=1)

    frac_lost = lost_energy / total_energy.clamp(min=1e-10)
    frac_recovered = recovered_energy / total_energy.clamp(min=1e-10)

    # Theoretical expectation for isotropic data
    theoretical = (d - k) / d

    print(f"  d={d}, k={k}, CR={d // k}")
    print(f"  Theoretical (isotropic): {theoretical:.4f} ({theoretical * 100:.2f}%)")
    print(f"  Actual frac_lost:   mean={frac_lost.mean():.4f}  "
          f"std={frac_lost.std():.4f}  "
          f"min={frac_lost.min():.4f}  max={frac_lost.max():.4f}")
    print(f"  Actual frac_recov:  mean={frac_recovered.mean():.4f}")

    result = {
        "experiment": "exp4_nullspace", "k": k, "cr": d // k,
        "d": d,
        "theoretical_frac_lost": theoretical,
        "frac_lost_mean": frac_lost.mean().item(),
        "frac_lost_std": frac_lost.std().item(),
        "frac_lost_min": frac_lost.min().item(),
        "frac_lost_max": frac_lost.max().item(),
        "frac_recovered_mean": frac_recovered.mean().item(),
    }
    return result


# ---------------------------------------------------------------------------
# Experiment 5: Per-Class Reconstruction Quality
# ---------------------------------------------------------------------------
def run_exp5_per_class(
    oracle: OracleActivationAttacker,
    act_decoder: ActivationDecoder,
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    k: int,
    d: int,
    dataset: str = "cifar10",
) -> List[Dict[str, float]]:
    """Exp 5: Per-class reconstruction quality using Exp 2's oracle decoder."""
    print("\n" + "=" * 70)
    print(f"  EXP 5: Per-Class Reconstruction Quality  (k={k})")
    print("=" * 70)

    num_classes = DATASET_STATS[dataset]["num_classes"]
    z_test = test_data["z"]
    x01_test = test_data["x01"]
    y_test = test_data["y"]

    oracle.eval()
    act_decoder.eval()

    # Reconstruct all test images
    all_recon = []
    bs = 256
    for i in range(0, z_test.size(0), bs):
        with torch.no_grad():
            h_hat = oracle(z_test[i:i + bs].to(device))
            recon = act_decoder(h_hat)
        all_recon.append(recon.cpu())
    recon_all = torch.cat(all_recon)

    # Per-class metrics
    results = []
    class_names = None
    if dataset == "cifar10":
        class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    print(f"  {'Class':>12s}  {'N':>5s}  {'MSE':>8s}  {'SSIM':>8s}  "
          f"{'PSNR':>8s}  {'LPIPS':>8s}")
    print(f"  {'-' * 12}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for c in range(num_classes):
        mask = (y_test == c)
        if mask.sum() == 0:
            continue
        orig_c = x01_test[mask]
        recon_c = recon_all[mask]
        m = compute_recon_metrics(recon_c, orig_c, device)

        name = class_names[c] if class_names else str(c)
        print(f"  {name:>12s}  {mask.sum():5d}  {m.mse:.6f}  {m.ssim:.4f}  "
              f"{m.psnr:8.2f}  {m.lpips:.4f}")

        results.append({
            "experiment": "exp5_per_class", "k": k, "cr": d // k,
            "class_id": c, "class_name": name,
            "n_samples": int(mask.sum()),
            "mse": m.mse, "ssim": m.ssim, "psnr": m.psnr, "lpips": m.lpips,
        })

    # Save per-class CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"per_class_metrics_k{k}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved to {csv_path}")

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Adaptive activations + gradients attacker
# ---------------------------------------------------------------------------
def run_exp6_grad_fusion(
    R: Optional[torch.Tensor],
    attacker_gdata: Dict[str, torch.Tensor],
    test_gdata: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    k: int,
    d: int,
    fuse_pinv: bool = False,
    attack_epochs: int = 200,
) -> Dict[str, float]:
    """Exp 6: adaptive attacker fusing activations z and the returned gradient g.

    Trains D(z, g[, z@R.T]) -> image on the auxiliary pool, evaluates on test.
    Sits next to Exp 3's activations-only D(z) row, isolating the marginal
    leakage from the returned gradients: it fuses all signals the server
    observes (projected activations and the returned gradient).
    """
    print("\n" + "=" * 70)
    tag = "z+g+pinv" if (fuse_pinv and R is not None) else "z+g"
    print(f"  EXP 6: Adaptive activations+gradients attacker  [{tag}]  (k={k})")
    print("=" * 70)

    # Standardize g per-dim using aux statistics; reapply the same transform at eval.
    g_aux = attacker_gdata["g"]
    g_mean = g_aux.mean(dim=0, keepdim=True)
    g_std = g_aux.std(dim=0, keepdim=True).clamp(min=1e-8)
    g_aux_n = (g_aux - g_mean) / g_std
    g_test_n = (test_gdata["g"] - g_mean) / g_std

    decoder = GradFusionImageDecoder(k, d, R=R, fuse_pinv=fuse_pinv)
    print(f"  Fusion decoder params: {count_parameters(decoder):,}")

    decoder = _train_image_decoder(
        decoder, attacker_gdata["z"], attacker_gdata["x01"],
        device, epochs=attack_epochs, label=f"GradFusion[{tag}]",
        train_g=g_aux_n,
    )

    # Evaluate on test (observed z and g)
    z_test = test_gdata["z"]
    x01_test = test_gdata["x01"]
    bs = 256
    recons = []
    decoder.eval()
    for i in range(0, z_test.size(0), bs):
        zb = z_test[i:i + bs].to(device)
        gb = g_test_n[i:i + bs].to(device)
        with torch.no_grad():
            recons.append(decoder(zb, gb).cpu())
    recon_all = torch.cat(recons)

    metrics = compute_recon_metrics(recon_all, x01_test, device)
    print(f"  Image-space:  MSE={metrics.mse:.6f}  SSIM={metrics.ssim:.4f}  "
          f"PSNR={metrics.psnr:.2f}  LPIPS={metrics.lpips:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(32, x01_test.size(0))
    grid = torch.cat([x01_test[:n_show], recon_all[:n_show]], dim=0)
    save_image(grid, out_dir / f"samples_k{k}_{tag}.png", nrow=n_show, padding=2)

    return {
        "experiment": "exp6_grad_fusion", "k": k, "cr": d // k,
        "fuse_pinv": bool(fuse_pinv and R is not None),
        "img_mse": metrics.mse, "img_ssim": metrics.ssim,
        "img_psnr": metrics.psnr, "img_lpips": metrics.lpips,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def _write_summary_csv(all_results, out_root):
    """Write summary.csv from the results accumulated so far.

    Called after every compression ratio (and once at the end) so a long or
    interrupted run leaves a usable CSV instead of nothing until completion.
    """
    if not all_results:
        return
    all_keys, seen = [], set()
    for r in all_results:
        for key in r.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)


def run_all_experiments(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = MODEL_CONFIGS[args.dataset]
    d = cfg["activation_dim"]  # 4096

    out_root = Path(args.output_dir)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data: deterministic 50/50 split — client pool vs attacker auxiliary pool.
    # The attacker NEVER sees the client's private data; it trains decoders on
    # auxiliary data from the same distribution (matching SDAR threat model).
    stats = DATASET_STATS[args.dataset]
    # No augmentation for activation collection (deterministic)
    collect_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    # Augmentation for split model training
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100

    # 90/10 split of training data: client gets 90%, attacker gets 10%
    full_train = cls(root=args.data_root, train=True, download=True, transform=collect_tf)
    n = len(full_train)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_attacker = n // 10                        # 5k samples
    client_pool = perm[: n - n_attacker]        # 45k samples — client's private data
    attacker_pool = perm[n - n_attacker :]      # 5k samples — attacker's auxiliary data

    # Client loader (with augmentation) for split model training
    train_aug = cls(root=args.data_root, train=True, download=False, transform=train_tf)
    client_train_loader = DataLoader(
        Subset(train_aug, client_pool),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    # Attacker auxiliary loader (no augmentation) for decoder training
    attacker_loader = DataLoader(
        Subset(full_train, attacker_pool),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    # Test set — used for evaluation only
    test_set = cls(root=args.data_root, train=False, download=True, transform=collect_tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"  Data split: {len(client_pool)} client / {len(attacker_pool)} attacker aux / "
          f"{len(test_set)} test")

    all_results = []
    mode = getattr(args, "mode", "fixed_random")

    cr_list = [1] if mode == "raw" else args.cr_list
    for cr in cr_list:
        k = 0 if mode == "raw" else d // cr
        print(f"\n{'#' * 70}")
        print(f"  COMPRESSION RATIO = {cr}x  (k={k}, d={d})")
        print(f"{'#' * 70}")

        # Phase A: get trained split model
        deep = getattr(args, "deep_head", False)
        rl = getattr(args, "recon_lambda", 0.0)
        wcc = getattr(args, "wcc_lambda", 0.0)
        tag = f"raw" if mode == "raw" else f"{mode}_k{k}"
        if mode == "block_random":
            # distinct block sizes share k but are different models -> distinct checkpoints
            tag += f"_b{getattr(args, 'block_size', 2)}"
        if deep:
            tag += "_deep"
        if rl > 0:
            tag += f"_rl{rl}"
        if wcc > 0:
            tag += f"_wcc{wcc}"
        ckpt_path = ckpt_dir / f"{tag}.pt"
        if args.load_split_checkpoint:
            ext_path = Path(args.load_split_checkpoint) / f"{tag}.pt"
            if ext_path.exists():
                ckpt_path = ext_path

        model = SplitModel(
            dataset=args.dataset, mode=mode, k=k,
            deep_head=getattr(args, "deep_head", False),
            clip_norm=getattr(args, "clip_norm", 0.0),
            noise_multiplier=getattr(args, "noise_multiplier", 0.0),
            block_size=getattr(args, "block_size", 2),
        ).to(device)

        if ckpt_path.exists():
            print(f"  Loading split model from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        else:
            print(f"  Training split model on CLIENT data only ({mode}, k={k})...")
            model = _train_split_model(
                model, client_train_loader, test_loader, device,
                epochs=args.split_epochs,
                patience=getattr(args, "split_patience", 100),
                recon_lambda=getattr(args, "recon_lambda", 0.0),
                wcc_lambda=getattr(args, "wcc_lambda", 0.0),
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

        R = getattr(model.bottleneck, "R", None)  # None for raw mode
        if mode == "block_random" and R is not None:
            # honest transmitted payload (n_blocks * out_per_block); equals d//cr for the
            # square 2^n grids we sweep, but use the true width so attack dims always match z
            k = R.shape[1]

        # Collect activations — attacker trains on auxiliary pool, evaluates on test
        print("  Collecting activations (attacker auxiliary pool)...")
        attacker_data = collect_activations(model, attacker_loader, device, args.dataset,
                                            max_samples=args.max_train_samples)
        print("  Collecting activations (test set)...")
        test_data = collect_activations(model, test_loader, device, args.dataset,
                                        max_samples=10000)
        print(f"  Attacker aux: {attacker_data['h'].shape}, Test: {test_data['h'].shape}")

        # Train ActivationDecoder (needed for Exp 1, 2, 5)
        act_decoder = None
        oracle = None
        exp2_recon = None

        needs_act_decoder = bool({0, 1, 2, 5} & set(args.experiments))
        if needs_act_decoder:
            act_dec_path = ckpt_dir / f"activation_decoder_{tag}.pt"
            act_decoder = ActivationDecoder(d)

            if act_dec_path.exists() and not args.retrain:
                print(f"  Loading ActivationDecoder from {act_dec_path}")
                act_decoder.load_state_dict(
                    torch.load(act_dec_path, map_location=device, weights_only=True))
                act_decoder = act_decoder.to(device).eval()
            else:
                print("  Training ActivationDecoder on auxiliary (h, x01) pairs...")
                act_decoder = _train_model_mse(
                    act_decoder, attacker_data["h"], attacker_data["x01"],
                    device, epochs=args.attack_epochs, lr=5e-4,
                    label="ActDecoder",
                )
                torch.save(act_decoder.state_dict(), act_dec_path)

        # --- Run experiments ---

        # Exp 0: Raw baseline (decoder on uncompressed activations)
        if 0 in args.experiments and act_decoder is not None:
            r0 = run_exp0_raw_baseline(
                act_decoder, test_data, device,
                out_root / "exp0_raw_baseline",
            )
            all_results.append(r0)

        if 4 in args.experiments and R is not None:
            r4 = run_exp4_nullspace(R, test_data, k, d)
            all_results.append(r4)

        if 1 in args.experiments and R is not None:
            r1 = run_exp1_pseudoinverse(
                R, act_decoder, test_data, device,
                out_root / "exp1_pseudoinverse", k,
            )
            all_results.append(r1)

        if 2 in args.experiments and k > 0:
            r2, oracle, exp2_recon = run_exp2_oracle(
                R, act_decoder, attacker_data, test_data, device,
                out_root / "exp2_oracle", k, d,
                attack_epochs=args.attack_epochs,
            )
            all_results.append(r2)

        if 3 in args.experiments and k > 0:
            r3 = run_exp3_known_vs_unknown(
                R, attacker_data, test_data, device,
                out_root / "exp3_known_vs_unknown", k, d,
                attack_epochs=args.attack_epochs,
            )
            all_results.append(r3)

        if 5 in args.experiments and R is not None:
            if oracle is None:
                # Need to train oracle if Exp 2 wasn't run
                print("  Exp 5 requires Oracle decoder — training...")
                oracle = OracleActivationAttacker(R, d, k)
                oracle = _train_model_mse(
                    oracle, attacker_data["z"], attacker_data["h"],
                    device, epochs=args.attack_epochs, lr=1e-3,
                    label="Oracle",
                )
            r5_list = run_exp5_per_class(
                oracle, act_decoder, test_data, device,
                out_root / "exp5_per_class", k, d, args.dataset,
            )
            all_results.extend(r5_list)

        if 6 in args.experiments and k > 0:
            print("  Collecting returned-gradient signals (aux + test) for Exp 6...")
            attacker_gdata = collect_activations_grads(
                model, attacker_loader, device, args.dataset,
                max_samples=args.max_train_samples)
            test_gdata = collect_activations_grads(
                model, test_loader, device, args.dataset, max_samples=10000)
            r6 = run_exp6_grad_fusion(
                R, attacker_gdata, test_gdata, device,
                out_root / "exp6_grad_fusion", k, d,
                fuse_pinv=args.fuse_pinv, attack_epochs=args.attack_epochs,
            )
            all_results.append(r6)

        # incremental save: rewrite summary.csv after every CR so a long or
        # interrupted run leaves usable results instead of nothing
        _write_summary_csv(all_results, out_root)
        print(f"  [summary.csv updated: {len(all_results)} rows through CR={cr}x]")

    # Final summary CSV (also written incrementally after each CR above)
    _write_summary_csv(all_results, out_root)
    if all_results:
        print(f"\nSummary saved to {out_root / 'summary.csv'}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in all_results:
        exp = r.get("experiment", "?")
        k_val = r.get("k", "?")
        cr_val = r.get("cr", "?")
        if "img_psnr" in r:
            print(f"  {exp:30s}  CR={cr_val:>3}  "
                  f"PSNR={r['img_psnr']:.2f}  SSIM={r['img_ssim']:.4f}  "
                  f"LPIPS={r.get('img_lpips', float('nan')):.4f}")
        elif "known_psnr" in r:
            print(f"  {exp:30s}  CR={cr_val:>3}  "
                  f"Known: PSNR={r['known_psnr']:.2f} SSIM={r['known_ssim']:.4f}  |  "
                  f"Unknown: PSNR={r['unknown_psnr']:.2f} SSIM={r['unknown_ssim']:.4f}")
        elif "frac_lost_mean" in r:
            print(f"  {exp:30s}  CR={cr_val:>3}  "
                  f"frac_lost={r['frac_lost_mean']:.4f}  "
                  f"theoretical={r['theoretical_frac_lost']:.4f}")
        elif "psnr" in r:
            print(f"  {exp:30s}  CR={cr_val:>3}  class={r.get('class_name', '?'):>12s}  "
                  f"PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Privileged attacker security analysis for fixed random projection")

    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--cr-list", nargs="+", type=int, default=[8, 16, 32],
                   help="Compression ratios to evaluate")
    p.add_argument("--mode", type=str, default="fixed_random",
                   choices=["raw", "fixed_random", "fixed_random_ll", "learned_1x1",
                            "block_random"],
                   help="Bottleneck mode")
    p.add_argument("--block-size", type=int, default=2,
                   help="block_random tile size: 1=per-position channel-only mix (max "
                        "utility, min privacy); cut_H (e.g. 8)=global dense R == "
                        "fixed_random (max privacy). Held-constant payload across sizes.")
    p.add_argument("--experiments", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                   help="Which experiments to run (0=raw baseline, 1-5; "
                        "6=adaptive activations+gradients fusion, opt-in)")
    p.add_argument("--split-epochs", type=int, default=2000,
                   help="Epochs for training the split model")
    p.add_argument("--split-patience", type=int, default=100,
                   help="Early stopping patience for split model (0 = disabled)")
    p.add_argument("--attack-epochs", type=int, default=200,
                   help="Epochs for training attack decoders")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-train-samples", type=int, default=50000,
                   help="Max training samples for activation collection")
    p.add_argument("--output-dir", default="./results/privileged_attack")
    p.add_argument("--load-split-checkpoint", default=None,
                   help="Directory with pre-trained split model checkpoints")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if checkpoints exist")
    p.add_argument("--deep-head", action="store_true",
                   help="Use deeper 2-block client head")
    # Exploratory; not reported in the paper.
    p.add_argument("--recon-lambda", type=float, default=0.0,
                   help="Reconstruction loss weight (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--clip-norm", type=float, default=0.0,
                   help="Per-sample L2 clip norm for activations (0 = disabled)")
    p.add_argument("--noise-multiplier", type=float, default=0.0,
                   help="Gaussian noise multiplier sigma (noise_std = sigma * clip_norm)")
    p.add_argument("--wcc-lambda", type=float, default=0.0,
                   help="Within-class compaction regularizer weight (0 = off)")
    p.add_argument("--fuse-pinv", action="store_true",
                   help="Exp 6: also feed the known-R pseudoinverse z@R.T into the "
                        "fusion decoder (strongest 'all signals' attacker)")

    return p.parse_args()


if __name__ == "__main__":
    run_all_experiments(parse_args())
