#!/usr/bin/env python3
"""SDAR (Simulator-based Data Reconstruction) attack for U-shaped split learning.

SDAR attacker adapted for this architecture:
  Head (client) -> Bottleneck -> [lift_back] -> Backbone (server) -> Tail (client)

Key differences from old VSL version:
  - ResNet18-based (64x8x8 head output, 512-dim backbone output)
  - Two cut points in fg_train_step (payload + backbone_out)
  - Non-conditional by default (U-shaped: server doesn't see client labels)
  - HeadSimulator (hs): server-side label estimator from backbone output
  - Supports learned_1x1 spatial bottleneck mode

Usage:
    python sdar_attack.py --mode raw --dataset cifar10
    python sdar_attack.py --mode fixed_random --compression-ratio 16
    python sdar_attack.py --mode learned_1x1 --compression-ratio 8 --deep-head
    python sdar_attack.py --sweep
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train_split import (
    SplitModel, Learned1x1Bottleneck, FixedRandomConvBottleneck,
    set_seed, DATASET_STATS, accuracy_from_logits, count_parameters,
    append_csv_row,
)
from client_models import MODEL_CONFIGS

# Spatial constants for ResNet18 head output
HEAD_C, HEAD_H, HEAD_W = 64, 8, 8


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def denormalize(x: torch.Tensor, dataset: str) -> torch.Tensor:
    """Undo channel-wise normalization, clamp to [0, 1]."""
    stats = DATASET_STATS[dataset]
    mean = torch.tensor(stats["mean"], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """Mean SSIM over a batch of [0,1] images (Gaussian-windowed)."""
    def _gauss_1d(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return (g / g.sum()).unsqueeze(0).unsqueeze(0)

    g1d = _gauss_1d(window_size).to(img1.device)
    window = (g1d.unsqueeze(-1) * g1d.unsqueeze(-2))  # (1,1,ws,ws)
    C = img1.shape[1]
    window = window.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Attack model architectures
# ---------------------------------------------------------------------------

class SimulatorEncoder(nn.Module):
    """Mimics client head + bottleneck (independent parameters).

    Output format matches the payload:
      raw:                spatial (B, 64, 8, 8)
      learned_1x1:        spatial (B, k_ch, 8, 8)
      fixed_random_conv:  spatial (B, k_ch, 8, 8)
      fixed_random:       flat (B, z_dim)
    """

    def __init__(self, mode: str, z_dim: int, k_channels: Optional[int] = None):
        super().__init__()
        self.mode = mode
        self.spatial = mode in ("raw", "learned_1x1", "fixed_random_conv")

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 32 -> 16
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 16 -> 8
            nn.Conv2d(64, HEAD_C, 3, padding=1),
            nn.BatchNorm2d(HEAD_C, momentum=0.1, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((HEAD_H, HEAD_W)),
        )

        if mode == "raw":
            pass  # output (B, 64, 8, 8) directly
        elif mode in ("learned_1x1", "fixed_random_conv"):
            self.compress = nn.Conv2d(HEAD_C, k_channels, 1)
        else:  # fixed_random
            self.proj = nn.Linear(HEAD_C * HEAD_H * HEAD_W, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # (B, 64, 8, 8)
        if self.mode == "raw":
            return feat
        elif self.mode in ("learned_1x1", "fixed_random_conv"):
            return self.compress(feat)
        else:
            return self.proj(feat.flatten(1))


class HeadSimulator(nn.Module):
    """Server-side label estimator: backbone_out (512-dim) -> num_classes."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class Decoder(nn.Module):
    """z -> 32x32 RGB.  Optionally label-conditional.

    Spatial input (raw/learned_1x1): (B, C, 8, 8) -> 2 upsamples -> 32x32
    Flat input (fixed_random):       (B, z_dim) -> FC -> (B, 256, 4, 4) -> 3 upsamples -> 32x32
    """

    def __init__(self, z_dim: int, num_classes: int = 10,
                 conditional: bool = False, spatial_input: bool = False,
                 payload_channels: int = 64):
        super().__init__()
        self.conditional = conditional
        self.spatial_input = spatial_input

        if spatial_input:
            c_in = payload_channels
            if conditional:
                self.label_embed = nn.Embedding(num_classes, HEAD_H * HEAD_W)
                c_in = payload_channels + 1

            self.net = nn.Sequential(
                nn.Conv2d(c_in, 256, 3, padding=1),
                nn.BatchNorm2d(256, momentum=0.1, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),                   # 8 -> 16
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128, momentum=0.1, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),                   # 16 -> 32
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64, momentum=0.1, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            if conditional:
                self.label_embed = nn.Embedding(num_classes, 50)
                self.label_proj = nn.Linear(50, z_dim)
                in_dim = z_dim * 2
            else:
                in_dim = z_dim

            self.fc = nn.Linear(in_dim, 256 * 4 * 4)
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=2),                   # 4 -> 8
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128, momentum=0.1, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),                   # 8 -> 16
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64, momentum=0.1, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),                   # 16 -> 32
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32, momentum=0.1, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(self, z: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.spatial_input:
            if self.conditional:
                y_emb = self.label_embed(y.long().view(-1))
                y_map = y_emb.view(-1, 1, HEAD_H, HEAD_W)
                z = torch.cat([z, y_map], dim=1)
            return self.net(z)
        else:
            if self.conditional:
                y_emb = self.label_proj(self.label_embed(y.long().view(-1)))
                z = torch.cat([z, y_emb], dim=1)
            return self.net(self.fc(z).view(-1, 256, 4, 4))


class SimulatorDiscriminator(nn.Module):
    """z -> real/fake.  Spatial or flat depending on mode."""

    def __init__(self, z_dim: int, num_classes: int = 10,
                 conditional: bool = False, spatial: bool = False,
                 payload_channels: int = 64):
        super().__init__()
        self.conditional = conditional
        self.spatial = spatial

        if spatial:
            c_in = payload_channels
            if conditional:
                self.label_embed = nn.Embedding(num_classes, HEAD_H * HEAD_W)
                c_in = payload_channels + 1
            self.net = nn.Sequential(
                nn.Conv2d(c_in, 128, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(64 * HEAD_H * HEAD_W, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1),
            )
        else:
            if conditional:
                self.label_embed = nn.Embedding(num_classes, 50)
                self.label_proj = nn.Linear(50, z_dim)
                in_dim = z_dim * 2
            else:
                in_dim = z_dim

            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1),
            )

    def forward(self, z: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.spatial:
            if self.conditional:
                y_emb = self.label_embed(y.long().view(-1))
                y_map = y_emb.view(-1, 1, HEAD_H, HEAD_W)
                z = torch.cat([z, y_map], dim=1)
            return self.net(z)
        else:
            if self.conditional:
                y_emb = self.label_proj(self.label_embed(y.long().view(-1)))
                z = torch.cat([z, y_emb], dim=1)
            return self.net(z)


class DecoderDiscriminator(nn.Module):
    """32x32 image -> real/fake.  Conv-based (DCGAN-style)."""

    def __init__(self, num_classes: int = 10, conditional: bool = False):
        super().__init__()
        self.conditional = conditional
        c_in = 3
        if conditional:
            self.label_embed = nn.Embedding(num_classes, 50)
            self.label_proj = nn.Linear(50, 32 * 32)
            c_in = 4

        self.net = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),       # 32 -> 16
            nn.BatchNorm2d(128, momentum=0.1, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),      # 16 -> 8
            nn.BatchNorm2d(128, momentum=0.1, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),      # 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.conditional:
            y_emb = self.label_proj(self.label_embed(y.long().view(-1)))
            x = torch.cat([x, y_emb.view(-1, 1, 32, 32)], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _dirichlet_partition(targets: np.ndarray, num_clients: int,
                         alpha: float, seed: int) -> List[List[int]]:
    """Dirichlet partition on a target array; returns per-client index lists."""
    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)
    buckets: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(props) * len(idx)).astype(int)
        splits = np.clip(splits, 0, len(idx))
        parts = np.split(idx, splits[:-1])
        for cid, part in enumerate(parts):
            buckets[cid].extend(part.tolist())
    return buckets


def get_attack_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    num_clients: int,
    alpha: float,
    seed: int,
    dataset: str,
) -> Tuple[List[DataLoader], DataLoader, List[DataLoader]]:
    """Create loaders for SDAR attack: 50% client, 50% server auxiliary.

    Returns (client_train_loaders, server_aux_loader, client_test_loaders).
    """
    stats = DATASET_STATS[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_set = cls(root=data_root, train=True, download=True, transform=train_tf)
    test_set = cls(root=data_root, train=False, download=True, transform=test_tf)

    # Deterministic 50/50 split of training data
    n = len(train_set)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    client_pool = perm[: n // 2]
    server_pool = perm[n // 2 :]

    # Dirichlet partition of the client pool among clients
    client_targets = np.array([train_set.targets[i] for i in client_pool])
    partition = _dirichlet_partition(client_targets, num_clients, alpha, seed)
    client_loaders: List[DataLoader] = []
    for local_idx in partition:
        orig_idx = [client_pool[i] for i in local_idx]
        client_loaders.append(
            DataLoader(
                Subset(train_set, orig_idx),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            )
        )

    # Server auxiliary loader
    server_loader = DataLoader(
        Subset(train_set, server_pool),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    # Per-client test loaders
    test_targets = np.array(test_set.targets)
    test_part = _dirichlet_partition(test_targets, num_clients, alpha, seed + 1)
    test_loaders: List[DataLoader] = []
    for idx in test_part:
        test_loaders.append(
            DataLoader(
                Subset(test_set, idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
        )

    return client_loaders, server_loader, test_loaders


# ---------------------------------------------------------------------------
# SDAR Attack
# ---------------------------------------------------------------------------

class SDARAttack:
    """Server-side SDAR reconstruction attack on U-shaped split learning."""

    def __init__(
        self,
        split_model: SplitModel,
        device: torch.device,
        dataset: str = "cifar10",
        conditional: bool = False,      # non-conditional by default (U-shaped)
        lambda1: float = 0.02,
        lambda2: float = 1e-5,
        flip_rate: float = 0.2,
        fg_lr: float = 1e-3,
        e_lr: float = 1e-3,
        d_lr: float = 5e-4,
        recon_lambda: float = 0.0,
        wcc_lambda: float = 0.0,
    ):
        self.model = split_model
        self.device = device
        self.dataset = dataset
        self.conditional = conditional
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.flip_rate = flip_rate
        self.recon_lambda = recon_lambda
        self.wcc_lambda = wcc_lambda

        cfg = MODEL_CONFIGS[dataset]
        num_classes = cfg["num_classes"]
        z_dim = split_model.payload_dim
        mode = split_model.mode
        spatial = mode in ("raw", "learned_1x1", "fixed_random_conv")

        # Payload channels for spatial modes
        if mode == "raw":
            payload_channels = HEAD_C  # 64
        elif mode in ("learned_1x1", "fixed_random_conv"):
            payload_channels = split_model._k_channels
        else:
            payload_channels = HEAD_C  # unused for flat modes

        k_channels = split_model._k_channels if mode in ("learned_1x1", "fixed_random_conv") else None

        # --- attack models ---
        self.e = SimulatorEncoder(mode, z_dim, k_channels=k_channels).to(device)
        self.d = Decoder(
            z_dim, num_classes, conditional,
            spatial_input=spatial,
            payload_channels=payload_channels,
        ).to(device)
        self.e_dis: Optional[SimulatorDiscriminator] = (
            SimulatorDiscriminator(
                z_dim, num_classes, conditional,
                spatial=spatial, payload_channels=payload_channels,
            ).to(device)
            if lambda1 > 0 else None
        )
        self.d_dis: Optional[DecoderDiscriminator] = (
            DecoderDiscriminator(num_classes, conditional).to(device)
            if lambda2 > 0 else None
        )

        # HeadSimulator: backbone_out -> estimated labels
        backbone_out_dim = cfg["backbone_out_dim"]  # 512
        self.hs = HeadSimulator(backbone_out_dim, num_classes).to(device)

        self.sim_lift = None

        # --- split-learning optimizers (client = head + bottleneck + tail) ---
        client_params = (
            list(split_model.head.parameters())
            + list(split_model.bottleneck.parameters())
            + list(split_model.tail.parameters())
        )
        self.fg_opt_client = torch.optim.Adam(
            client_params, lr=fg_lr, weight_decay=1e-4,
        )
        self.fg_opt_server = torch.optim.Adam(
            split_model.backbone.parameters(), lr=fg_lr, weight_decay=1e-4,
        )

        # --- attack optimizers ---
        e_params = list(self.e.parameters())
        if self.sim_lift is not None:
            e_params += list(self.sim_lift.parameters())
        self.e_opt = torch.optim.Adam(e_params, lr=e_lr)
        self.d_opt = torch.optim.Adam(self.d.parameters(), lr=d_lr)
        self.hs_opt = torch.optim.Adam(self.hs.parameters(), lr=e_lr)

        if self.e_dis is not None:
            self.e_dis_opt = torch.optim.Adam(
                self.e_dis.parameters(), lr=e_lr * lambda1,
            )
        if self.d_dis is not None:
            self.d_dis_opt = torch.optim.Adam(
                self.d_dis.parameters(), lr=d_lr * lambda2,
            )

    # ---- helpers -----------------------------------------------------------

    def _lift_back(self, payload: torch.Tensor,
                   info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Route real payload through lift_back -> spatial (B, 64, 8, 8)."""
        if isinstance(self.model.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
            return self.model.bottleneck.lift_back(payload, info)
        else:
            z_flat = self.model.bottleneck.lift_back(payload, info)
            return z_flat.view(-1, HEAD_C, HEAD_H, HEAD_W)

    def _format_payload(self, payload: torch.Tensor) -> torch.Tensor:
        """Reshape payload to match SimulatorEncoder output format.

        Raw mode payload is flat (B, 4096) but decoder/discriminator expect
        spatial (B, 64, 8, 8).  Other modes are already in the right format.
        """
        if self.model.mode == "raw" and payload.dim() == 2:
            return payload.view(-1, HEAD_C, HEAD_H, HEAD_W)
        return payload

    def _sim_server_forward(self, zs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route simulator output through server pipeline -> (logits, backbone_out).

        Handles lift-back differently per mode since simulator doesn't have
        client-side info.
        """
        mode = self.model.mode
        if mode == "raw":
            z_approx = zs  # already (B, 64, 8, 8)
        elif mode in ("learned_1x1", "fixed_random_conv"):
            z_approx = self.model.bottleneck.lift_back(zs, {})
        elif mode in ("fixed_random", "fixed_random_ll"):
            z_flat = zs @ self.model.bottleneck.R.T
            z_approx = z_flat.view(-1, HEAD_C, HEAD_H, HEAD_W)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        backbone_out = self.model.backbone(z_approx)
        logits = self.model.tail(backbone_out)
        return logits, backbone_out

    # ---- step 1: U-shaped split learning (two cut points) -----------------

    def fg_train_step(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> Tuple[float, torch.Tensor, Dict[str, torch.Tensor], float]:
        """Standard U-shaped split-learning forward/backward.

        Returns (loss, payload, info, acc).
        """
        self.model.train()

        # --- client forward: head -> bottleneck ---
        z_spatial = self.model.head(x)
        if isinstance(self.model.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
            payload, info = self.model.bottleneck(z_spatial)
        else:
            z = z_spatial.flatten(1)
            payload, info = self.model.bottleneck(z)

        # --- cut 1: client -> server (payload) ---
        payload_server = payload.detach().requires_grad_(True)

        # --- server forward: lift_back -> backbone ---
        z_approx = self._lift_back(payload_server, info)
        backbone_out = self.model.backbone(z_approx)

        # --- cut 2: server -> client (backbone_out) ---
        backbone_client = backbone_out.detach().requires_grad_(True)

        # --- client forward: tail -> logits -> loss ---
        self.fg_opt_client.zero_grad(set_to_none=True)
        logits = self.model.tail(backbone_client)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # --- backward phase: tail -> backbone -> head+bottleneck ---
        grad_backbone = backbone_client.grad.detach().clone()

        self.fg_opt_server.zero_grad(set_to_none=True)
        if self.recon_lambda > 0:
            z_target = z_spatial.detach()
            if not isinstance(self.model.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
                z_target = z_target.flatten(1)
            recon_loss = self.recon_lambda * F.mse_loss(
                z_approx, z_target.reshape_as(z_approx))
            backbone_out.backward(grad_backbone, retain_graph=True)
            recon_loss.backward()
        else:
            backbone_out.backward(grad_backbone)
        torch.nn.utils.clip_grad_norm_(
            self.model.backbone.parameters(), max_norm=1.0)
        self.fg_opt_server.step()

        grad_payload = payload_server.grad.detach().clone()

        # WCC loss: added to client backward pass (backprops through head+bottleneck)
        wcc_lambda = getattr(self, "wcc_lambda", 0.0)
        if wcc_lambda > 0 and "bottleneck_z" in info:
            z_bn = info["bottleneck_z"]
            wcc_loss = torch.tensor(0.0, device=payload.device)
            for c in y.unique():
                mask = (y == c)
                if mask.sum() > 1:
                    z_c = z_bn[mask]
                    mu_c = z_c.mean(dim=0, keepdim=True)
                    wcc_loss = wcc_loss + (z_c - mu_c).pow(2).sum(dim=1).mean()
            payload.backward(grad_payload, retain_graph=True)
            (wcc_lambda * wcc_loss).backward()
        else:
            payload.backward(grad_payload)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.head.parameters())
            + list(self.model.bottleneck.parameters())
            + list(self.model.tail.parameters()),
            max_norm=1.0,
        )
        self.fg_opt_client.step()

        acc = accuracy_from_logits(logits, y) / y.size(0)

        # Return z_real in the format matching SimulatorEncoder output
        z_real = payload.detach()
        if self.model.mode == "raw":
            z_real = z_real.view(-1, HEAD_C, HEAD_H, HEAD_W)
        return loss.item(), z_real, info, acc

    # ---- step 2: train simulator e (+ discriminator + hs) -----------------

    def e_train_step(
        self,
        xs: torch.Tensor, ys: torch.Tensor,
        y: torch.Tensor, z_real: torch.Tensor,
    ) -> Dict[str, float]:
        """Train simulator encoder e, its discriminator, and head simulator hs."""
        metrics: Dict[str, float] = {}

        # -- 2a: update simulator discriminator (if present) ----------------
        if self.e_dis is not None:
            self.e_dis.train()
            self.e.eval()
            with torch.no_grad():
                zs_det = self.e(xs)

            if self.conditional:
                real_out = self.e_dis(z_real, y)
                fake_out = self.e_dis(zs_det, ys)
            else:
                real_out = self.e_dis(z_real)
                fake_out = self.e_dis(zs_det)

            e_dis_loss = (
                F.binary_cross_entropy_with_logits(
                    real_out, torch.ones_like(real_out))
                + F.binary_cross_entropy_with_logits(
                    fake_out, torch.zeros_like(fake_out))
            )
            self.e_dis_opt.zero_grad(set_to_none=True)
            e_dis_loss.backward()
            self.e_dis_opt.step()
            metrics["e_dis_loss"] = e_dis_loss.item()

        # -- 2b: update simulator encoder -----------------------------------
        self.e.train()
        self.model.backbone.eval()
        self.model.tail.eval()

        zs = self.e(xs)
        logits_sim, backbone_out_sim = self._sim_server_forward(zs)
        eg_loss = F.cross_entropy(logits_sim, ys)
        eg_acc = accuracy_from_logits(logits_sim, ys) / ys.size(0)

        e_total = eg_loss
        if self.e_dis is not None:
            self.e_dis.eval()
            if self.conditional:
                fake_out = self.e_dis(zs, ys)
            else:
                fake_out = self.e_dis(zs)
            e_gen_loss = F.binary_cross_entropy_with_logits(
                fake_out, torch.ones_like(fake_out))
            e_total = eg_loss + e_gen_loss * self.lambda1
            metrics["e_gen_loss"] = e_gen_loss.item()

        self.e_opt.zero_grad(set_to_none=True)
        e_total.backward()
        self.e_opt.step()

        # -- 2c: train head simulator (hs) on backbone output ---------------
        self.hs.train()
        hs_logits = self.hs(backbone_out_sim.detach())
        hs_loss = F.cross_entropy(hs_logits, ys)
        hs_acc = accuracy_from_logits(hs_logits, ys) / ys.size(0)

        self.hs_opt.zero_grad(set_to_none=True)
        hs_loss.backward()
        self.hs_opt.step()

        self.model.backbone.train()
        self.model.tail.train()
        metrics["eg_loss"] = eg_loss.item()
        metrics["eg_acc"] = eg_acc
        metrics["hs_loss"] = hs_loss.item()
        metrics["hs_acc"] = hs_acc
        return metrics

    # ---- step 3: train decoder d (+ discriminator) ------------------------

    def d_train_step(
        self,
        xs: torch.Tensor, ys: torch.Tensor,
        xs_raw: torch.Tensor,
        y: torch.Tensor, z_real: torch.Tensor,
    ) -> Dict[str, float]:
        """Train decoder d and its discriminator.

        xs     : normalised server images  (simulator input)
        xs_raw : [0,1] server images       (decoder reconstruction target)
        """
        metrics: Dict[str, float] = {}

        # -- 3a: update decoder discriminator (if present) ------------------
        if self.d_dis is not None:
            self.d_dis.train()
            self.d.eval()
            with torch.no_grad():
                if self.conditional:
                    decoded_z = self.d(z_real, y)
                else:
                    decoded_z = self.d(z_real)

            if self.conditional:
                real_out = self.d_dis(xs_raw, ys)
                fake_out = self.d_dis(decoded_z, y)
            else:
                real_out = self.d_dis(xs_raw)
                fake_out = self.d_dis(decoded_z)

            d_dis_loss = (
                F.binary_cross_entropy_with_logits(
                    real_out, torch.ones_like(real_out))
                + F.binary_cross_entropy_with_logits(
                    fake_out, torch.zeros_like(fake_out))
            )
            self.d_dis_opt.zero_grad(set_to_none=True)
            d_dis_loss.backward()
            self.d_dis_opt.step()
            metrics["d_dis_loss"] = d_dis_loss.item()

        # -- 3b: update decoder ---------------------------------------------
        self.d.train()
        self.e.eval()

        with torch.no_grad():
            zs = self.e(xs)

        if self.conditional:
            decoded_xs = self.d(zs, ys)
        else:
            decoded_xs = self.d(zs)

        d_mse = F.mse_loss(decoded_xs, xs_raw)
        d_total = d_mse

        if self.d_dis is not None:
            self.d_dis.eval()
            if self.conditional:
                decoded_z = self.d(z_real, y)
                fake_out = self.d_dis(decoded_z, y)
            else:
                decoded_z = self.d(z_real)
                fake_out = self.d_dis(decoded_z)
            d_gen_loss = F.binary_cross_entropy_with_logits(
                fake_out, torch.ones_like(fake_out))
            d_total = d_mse + d_gen_loss * self.lambda2
            metrics["d_gen_loss"] = d_gen_loss.item()

        self.d_opt.zero_grad(set_to_none=True)
        d_total.backward()
        self.d_opt.step()

        metrics["d_mse"] = d_mse.item()
        return metrics

    # ---- evaluation -------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """MSE + SSIM on a client's private data (ground-truth comparison)."""
        self.model.eval()
        self.d.eval()
        self.hs.eval()
        total_mse, total_ssim, total_hs_correct, count, total_samples = (
            0.0, 0.0, 0, 0, 0,
        )

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Get payload from real client pipeline
            z_spatial = self.model.head(x)
            if isinstance(self.model.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
                payload, info = self.model.bottleneck(z_spatial)
            else:
                z = z_spatial.flatten(1)
                payload, info = self.model.bottleneck(z)

            # Reconstruct (reshape payload for raw mode)
            z_dec = self._format_payload(payload)
            recon = self.d(z_dec, y) if self.conditional else self.d(z_dec)

            x01 = denormalize(x, self.dataset)
            total_mse += F.mse_loss(recon, x01).item()
            total_ssim += compute_ssim(recon, x01)

            # hs label accuracy (route through backbone)
            z_approx = self._lift_back(payload, info)
            backbone_out = self.model.backbone(z_approx)
            hs_pred = self.hs(backbone_out).argmax(dim=1)
            total_hs_correct += (hs_pred == y).sum().item()
            total_samples += y.size(0)
            count += 1

        n = max(count, 1)
        return {
            "mse": total_mse / n,
            "ssim": total_ssim / n,
            "hs_acc": total_hs_correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def save_samples(self, loader: DataLoader, out_dir: Path,
                     n: int = 32) -> None:
        """Save original vs reconstructed image grids."""
        self.model.eval()
        self.d.eval()
        out_dir.mkdir(parents=True, exist_ok=True)

        origs, recons = [], []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            z_spatial = self.model.head(x)
            if isinstance(self.model.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
                payload, info = self.model.bottleneck(z_spatial)
            else:
                z = z_spatial.flatten(1)
                payload, info = self.model.bottleneck(z)

            z_dec = self._format_payload(payload)
            recon = self.d(z_dec, y) if self.conditional else self.d(z_dec)
            origs.append(denormalize(x, self.dataset))
            recons.append(recon)
            if sum(o.size(0) for o in origs) >= n:
                break

        orig = torch.cat(origs)[:n]
        recon = torch.cat(recons)[:n]
        grid = torch.cat([orig, recon], dim=0)
        save_image(grid, out_dir / "comparison.png", nrow=n, padding=2)
        save_image(orig, out_dir / "originals.png", nrow=8, padding=2)
        save_image(recon, out_dir / "reconstructions.png", nrow=8, padding=2)

    # ---- main training loop -----------------------------------------------

    def run(
        self,
        client_loaders: List[DataLoader],
        server_loader: DataLoader,
        test_loaders: List[DataLoader],
        num_iters: int = 20000,
        eval_every: int = 500,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """Combined split-learning training + SDAR attack."""
        num_classes = DATASET_STATS[self.dataset]["num_classes"]
        log_keys = [
            "fg_loss", "fg_acc",
            "eg_loss", "eg_acc", "e_gen_loss", "e_dis_loss",
            "d_mse", "d_gen_loss", "d_dis_loss",
            "hs_loss", "hs_acc",
            "attack_mse",
        ]
        log = {k: np.zeros(num_iters) for k in log_keys}

        client_iters = [iter(_InfLoader(ld)) for ld in client_loaders]
        server_iter = iter(_InfLoader(server_loader))
        num_clients = len(client_loaders)

        for i in range(num_iters):
            # pick client round-robin
            cid = i % num_clients
            x, y = next(client_iters[cid])
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            xs, ys = next(server_iter)
            xs = xs.to(self.device, non_blocking=True)
            ys = ys.to(self.device, non_blocking=True)

            # optional label flip on auxiliary data
            if self.flip_rate > 0:
                mask = torch.rand(ys.size(), device=self.device) < self.flip_rate
                ys = torch.where(
                    mask,
                    torch.randint_like(ys, 0, num_classes),
                    ys,
                )

            # denormalised server images (decoder target)
            xs_raw = denormalize(xs, self.dataset)

            # step 1: split learning
            fg_loss, z_real, info, fg_acc = self.fg_train_step(x, y)
            log["fg_loss"][i] = fg_loss
            log["fg_acc"][i] = fg_acc

            # step 2: train simulator
            e_met = self.e_train_step(xs, ys, y, z_real)
            for k, v in e_met.items():
                if k in log:
                    log[k][i] = v

            # step 3: train decoder
            d_met = self.d_train_step(xs, ys, xs_raw, y, z_real)
            for k, v in d_met.items():
                if k in log:
                    log[k][i] = v

            # instantaneous attack MSE on this batch
            self.d.eval()
            with torch.no_grad():
                x01 = denormalize(x, self.dataset)
                recon = (self.d(z_real, y) if self.conditional
                         else self.d(z_real))
                log["attack_mse"][i] = F.mse_loss(recon, x01).item()

            # periodic logging + eval
            if (i + 1) % eval_every == 0:
                window = slice(max(0, i + 1 - eval_every), i + 1)
                print(
                    f"[{i+1:>6d}/{num_iters}]  "
                    f"fg_loss {log['fg_loss'][window].mean():.4f}  "
                    f"fg_acc {log['fg_acc'][window].mean():.3f}  "
                    f"eg_loss {log['eg_loss'][window].mean():.4f}  "
                    f"d_mse {log['d_mse'][window].mean():.4f}  "
                    f"atk_mse {log['attack_mse'][window].mean():.4f}  "
                    f"hs_acc {log['hs_acc'][window].mean():.3f}"
                )
                ev = self.evaluate(test_loaders[0])
                print(
                    f"         eval  mse {ev['mse']:.4f}  "
                    f"ssim {ev['ssim']:.4f}  hs_acc {ev['hs_acc']:.4f}"
                )
                if save_dir is not None:
                    self.save_samples(
                        test_loaders[0],
                        save_dir / f"iter_{i+1:06d}",
                    )

        return log


class _InfLoader:
    """Wraps a DataLoader so it cycles forever."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class AttackResult:
    dataset: str
    mode: str
    k: int
    deep_head: bool
    recon_lambda: float
    num_clients: int
    alpha: float
    conditional: bool
    attack_iters: int
    lambda1: float
    lambda2: float
    flip_rate: float
    final_mse: float
    final_ssim: float
    final_hs_acc: float
    final_fg_acc: float


def run_attack_experiment(
    args,
    mode: str,
    k: int,
    client_loaders: List[DataLoader],
    server_loader: DataLoader,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> AttackResult:
    """Run a single SDAR attack experiment for one bottleneck mode."""
    cfg = MODEL_CONFIGS[args.dataset]

    model = SplitModel(
        dataset=args.dataset,
        mode=mode,
        k=k,
        deep_head=args.deep_head,
        conv_kernel_size=getattr(args, "conv_kernel_size", 1),
        clip_norm=getattr(args, "clip_norm", 0.0),
        noise_multiplier=getattr(args, "noise_multiplier", 0.0),
    ).to(device)

    ks_tag = f"_ks{args.conv_kernel_size}" if getattr(args, "conv_kernel_size", 1) > 1 else ""
    save_dir = Path(args.out_dir) / f"{args.dataset}_{mode}_k{k}{ks_tag}"
    spatial = mode in ("raw", "learned_1x1", "fixed_random_conv")

    print("=" * 80)
    print(f"SDAR Attack  |  mode={mode}  k={k}  payload={model.payload_dim}"
          f"  deep_head={args.deep_head}")
    print(f"  dataset={args.dataset}  clients={len(client_loaders)}  "
          f"alpha={args.alpha}  iters={args.attack_iters}")
    print(f"  conditional={args.conditional}  lambda1={args.lambda1}  "
          f"lambda2={args.lambda2}  flip_rate={args.flip_rate}")
    print(f"  head params:     {count_parameters(model.head):,}")
    print(f"  backbone params: {count_parameters(model.backbone):,}")
    print(f"  tail params:     {count_parameters(model.tail):,}")
    e_tmp = SimulatorEncoder(
        mode, model.payload_dim,
        k_channels=getattr(model, "_k_channels", None),
    )
    print(f"  simulator (e) params: {count_parameters(e_tmp):,}")
    del e_tmp
    print("=" * 80)

    attacker = SDARAttack(
        split_model=model,
        device=device,
        dataset=args.dataset,
        conditional=args.conditional,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        flip_rate=args.flip_rate,
        fg_lr=args.fg_lr,
        e_lr=args.e_lr,
        d_lr=args.d_lr,
        recon_lambda=getattr(args, "recon_lambda", 0.0),
        wcc_lambda=getattr(args, "wcc_lambda", 0.0),
    )

    log = attacker.run(
        client_loaders=client_loaders,
        server_loader=server_loader,
        test_loaders=test_loaders,
        num_iters=args.attack_iters,
        eval_every=args.eval_every,
        save_dir=save_dir,
    )

    # final evaluation on all client test data
    all_mse, all_ssim, all_hs, n_loaders = 0.0, 0.0, 0.0, 0
    for tl in test_loaders:
        ev = attacker.evaluate(tl)
        all_mse += ev["mse"]
        all_ssim += ev["ssim"]
        all_hs += ev["hs_acc"]
        n_loaders += 1
    final_mse = all_mse / max(n_loaders, 1)
    final_ssim = all_ssim / max(n_loaders, 1)
    final_hs_acc = all_hs / max(n_loaders, 1)

    tail = max(1, args.attack_iters // 10)
    final_fg_acc = float(log["fg_acc"][-tail:].mean())

    print("-" * 80)
    print(f"FINAL  mode={mode}  k={k}")
    print(f"  MSE    = {final_mse:.6f}")
    print(f"  SSIM   = {final_ssim:.4f}")
    print(f"  hs_acc = {final_hs_acc:.4f}")
    print(f"  SL accuracy (last 10%) = {final_fg_acc:.4f}")
    print()

    attacker.save_samples(test_loaders[0], save_dir / "final", n=64)
    np.savez(save_dir / "log.npz", **log)

    result = AttackResult(
        dataset=args.dataset, mode=mode, k=k,
        deep_head=args.deep_head,
        recon_lambda=getattr(args, "recon_lambda", 0.0),
        num_clients=len(client_loaders), alpha=args.alpha,
        conditional=args.conditional,
        attack_iters=args.attack_iters,
        lambda1=args.lambda1, lambda2=args.lambda2,
        flip_rate=args.flip_rate,
        final_mse=final_mse, final_ssim=final_ssim,
        final_hs_acc=final_hs_acc, final_fg_acc=final_fg_acc,
    )

    if args.results_csv:
        append_csv_row(Path(args.results_csv), asdict(result))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SDAR reconstruction attack on U-shaped split learning")

    # data / model
    p.add_argument("--data-root", default="./data")
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--mode", default="raw",
                   choices=["raw", "fixed_random", "fixed_random_ll", "fixed_random_conv", "learned_1x1"])
    p.add_argument("--compression-ratio", type=int, default=16,
                   help="Compression ratio (d/k). k = activation_dim // CR")
    p.add_argument("--conv-kernel-size", type=int, default=1,
                   help="Kernel size for fixed_random_conv (1=channel-only, 3=spatial mixing)")
    p.add_argument("--deep-head", action="store_true",
                   help="Use deeper 2-block head (~46.5K params)")
    p.add_argument("--recon-lambda", type=float, default=0.0,
                   help="Reconstruction loss weight for foreground training (0=off)")

    # multi-client
    p.add_argument("--num-clients", type=int, default=1)
    p.add_argument("--alpha", type=float, default=10000000.0,
                   help="Dirichlet alpha (higher = more IID)")

    # attack hyper-parameters (USL defaults from original SDAR paper)
    p.add_argument("--attack-iters", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--fg-lr", type=float, default=1e-3)
    p.add_argument("--e-lr", type=float, default=1e-3)
    p.add_argument("--d-lr", type=float, default=5e-4)
    p.add_argument("--lambda1", type=float, default=0.02,
                   help="Weight for simulator discriminator GAN loss")
    p.add_argument("--lambda2", type=float, default=1e-5,
                   help="Weight for decoder discriminator GAN loss")
    p.add_argument("--flip-rate", type=float, default=0.2,
                   help="Fraction of auxiliary labels to randomly flip (USL default)")
    p.add_argument("--conditional", action="store_true",
                   help="Use conditional decoder (VSL mode). Default: non-conditional (USL)")
    p.add_argument("--eval-every", type=int, default=500)

    # sweep
    p.add_argument("--sweep", action="store_true",
                   help="Evaluate all bottleneck modes")
    p.add_argument("--modes", nargs="+",
                   default=["raw", "fixed_random", "fixed_random_conv", "learned_1x1"])
    p.add_argument("--cr-list", nargs="+", type=int, default=[8, 16, 32],
                   help="Compression ratios to sweep")

    # output
    p.add_argument("--out-dir", default="./sdar_results")
    p.add_argument("--results-csv", default="sdar_attack_results.csv")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gpu", type=int, default=None)

    # differential privacy
    p.add_argument("--clip-norm", type=float, default=0.0,
                   help="Per-sample L2 clip norm for activations (0 = disabled)")
    p.add_argument("--noise-multiplier", type=float, default=0.0,
                   help="Gaussian noise multiplier sigma (noise_std = sigma * clip_norm)")
    p.add_argument("--wcc-lambda", type=float, default=0.0,
                   help="Within-class compaction regularizer weight (0 = off)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_loaders, server_loader, test_loaders = get_attack_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        dataset=args.dataset,
    )

    results: List[AttackResult] = []
    activation_dim = MODEL_CONFIGS[args.dataset]["activation_dim"]

    if args.sweep:
        for mode in args.modes:
            if mode == "raw":
                results.append(run_attack_experiment(
                    args, "raw", 0,
                    client_loaders, server_loader, test_loaders, device))
            else:
                for cr in args.cr_list:
                    k = activation_dim // cr
                    results.append(run_attack_experiment(
                        args, mode, k,
                        client_loaders, server_loader, test_loaders, device))
    else:
        k = 0 if args.mode == "raw" else activation_dim // args.compression_ratio
        results.append(run_attack_experiment(
            args, args.mode, k,
            client_loaders, server_loader, test_loaders, device))

    # summary
    print("\n" + "=" * 80)
    print("SDAR ATTACK SUMMARY")
    print("=" * 80)
    print(f"{'mode':<18s} {'k':>4s} {'MSE':>10s} {'SSIM':>8s} "
          f"{'hs_acc':>8s} {'SL_acc':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r.mode:<18s} {r.k:>4d} "
              f"{r.final_mse:>10.6f} {r.final_ssim:>8.4f} "
              f"{r.final_hs_acc:>8.4f} {r.final_fg_acc:>8.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
