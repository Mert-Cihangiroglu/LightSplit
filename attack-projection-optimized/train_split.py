#!/usr/bin/env python3
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from client_models import build_head, build_backbone, build_tail, MODEL_CONFIGS


class RawBottleneck(nn.Module):
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return h, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload








class FixedRandomBottleneck(nn.Module):
    def __init__(self, activation_dim: int, k: int,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        info = {}

        # Per-sample L2 clipping (differentiable, bounds sensitivity to 2C)
        if self.clip_norm > 0:
            norms = h.norm(dim=1, keepdim=True)  # (B, 1)
            clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
            h = h * clip_factor
            info["clip_fraction"] = (norms.squeeze() > self.clip_norm).float().mean().detach()
            info["activation_norm_mean"] = norms.mean().detach()

        z = h @ self.R

        # Gaussian noise (train + eval — required for formal DP guarantee)
        if self.noise_multiplier > 0 and self.clip_norm > 0:
            noise_std = self.noise_multiplier * self.clip_norm
            z = z + torch.randn_like(z) * noise_std
            info["noise_std"] = noise_std

        # Store z for within-class compaction loss
        info["bottleneck_z"] = z

        return z, info

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload @ self.R.T


class BlockRandomBottleneck(nn.Module):
    """Block-local fixed orthonormal random projection: a privacy/utility dial.

    Tiles the (C,H,W) cut into block x block spatial tiles (each tile = C*block*block
    activations spanning ALL channels), projects every tile with ONE shared orthonormal
    random basis R_b (in_per_block -> out_per_block), and lifts back with R_b^T.  The
    macro grid (H/block x W/block) is preserved, so the random mixing is confined to a
    block x block receptive field -- the knob between privacy and utility:
        block = 1   -> per-position channel-only mix (max utility, min privacy)
        block = H   -> single global dense R == FixedRandomBottleneck (max privacy)

    For the white-box (known-R) attacker we expose the FULL effective projection as a
    single buffer `R` of shape (activation_dim, payload): a permuted block-diagonal of
    R_b whose columns are orthonormal.  The privileged repo flattens the cut as
    h = z_spatial.flatten(1) (channels-first: flat index = c*H*W + y*W + x), so `R`
    scatters each tile's R_b into exactly the flat rows that tile owns.  Then
    z = h @ R reproduces the tile-wise projection EXACTLY, the lift z @ R.T = P h is the
    rank-`payload` orthogonal projector, and every known-R attack (pseudoinverse, oracle
    residual, known-R decoder, null-space energy, gradient fusion) reuses
    model.bottleneck.R unchanged -- block_random is just fixed_random with a
    block-structured R.  Numerically identical to the unfold-based PyTorch
    BlockRandomBottleneck in utility/train_split.py (and the TF port in SDAR); the
    flat-R form is used here so z and R are guaranteed consistent for the attacker.
    R is a client-only constant (never trained, never transmitted).
    """

    def __init__(self, channels: int, hw_h: int, hw_w: int, block: int, k: int,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0):
        super().__init__()
        if block < 1 or hw_h % block != 0 or hw_w % block != 0:
            raise ValueError(
                f"block={block} must be >=1 and divide spatial dims ({hw_h},{hw_w}).")
        self.channels, self.h, self.w, self.block = channels, hw_h, hw_w, block
        self.grid_h, self.grid_w = hw_h // block, hw_w // block
        self.n_blocks = self.grid_h * self.grid_w
        self.in_per_block = channels * block * block
        # floor-divide the float budget across tiles (exact for square 2^n grids); the
        # honest transmitted size is payload_floats = n_blocks * out_per_block.
        self.out_per_block = max(1, k // self.n_blocks)
        self.payload_floats = self.n_blocks * self.out_per_block
        self.activation_dim = channels * hw_h * hw_w
        self.k = self.payload_floats
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

        # One shared orthonormal basis per tile (in_per_block -> out_per_block).
        rand_matrix = torch.randn(self.in_per_block, self.out_per_block)
        q, _ = torch.linalg.qr(rand_matrix)            # orthonormal columns

        # Assemble the full (activation_dim, payload) projection R by scattering R_b into
        # the flat rows owned by each tile.  Within-tile element q decodes channel-major
        # to (c, by, bx) == (q // block^2, (q % block^2) // block, q % block), matching the
        # unfold ordering (channel -> block_row -> block_col); it sits at spatial position
        # (gh*block + by, gw*block + bx) for tile t = gh*grid_w + gw, hence flat row
        # c*H*W + y*W + x.  Disjoint tiles touch disjoint rows, so R's columns stay
        # orthonormal and R.T @ R = I.
        R_full = torch.zeros(self.activation_dim, self.payload_floats)
        bb = block * block
        q_idx = torch.arange(self.in_per_block)
        c = q_idx // bb
        by = (q_idx % bb) // block
        bx = q_idx % block
        for t in range(self.n_blocks):
            gh, gw = t // self.grid_w, t % self.grid_w
            rows = c * (hw_h * hw_w) + (gh * block + by) * hw_w + (gw * block + bx)
            cols = torch.arange(t * self.out_per_block, (t + 1) * self.out_per_block)
            R_full[rows.unsqueeze(1), cols.unsqueeze(0)] = q
        self.register_buffer("R", R_full)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        info = {}

        # Per-sample L2 clipping (differentiable, bounds sensitivity to 2C)
        if self.clip_norm > 0:
            norms = h.norm(dim=1, keepdim=True)
            clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
            h = h * clip_factor
            info["clip_fraction"] = (norms.squeeze() > self.clip_norm).float().mean().detach()
            info["activation_norm_mean"] = norms.mean().detach()

        z = h @ self.R

        if self.noise_multiplier > 0 and self.clip_norm > 0:
            noise_std = self.noise_multiplier * self.clip_norm
            z = z + torch.randn_like(z) * noise_std
            info["noise_std"] = noise_std

        info["bottleneck_z"] = z
        return z, info

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload @ self.R.T


class FixedRandomResidualBottleneck(nn.Module):
    def __init__(self, activation_dim: int, k: int,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        info = {}

        # Per-sample L2 clipping (differentiable, bounds sensitivity to 2C)
        if self.clip_norm > 0:
            norms = h.norm(dim=1, keepdim=True)  # (B, 1)
            clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
            h = h * clip_factor
            info["clip_fraction"] = (norms.squeeze() > self.clip_norm).float().mean().detach()
            info["activation_norm_mean"] = norms.mean().detach()

        z = h @ self.R

        # Gaussian noise (train + eval — required for formal DP guarantee)
        if self.noise_multiplier > 0 and self.clip_norm > 0:
            noise_std = self.noise_multiplier * self.clip_norm
            z = z + torch.randn_like(z) * noise_std
            info["noise_std"] = noise_std

        # Store z for within-class compaction loss
        info["bottleneck_z"] = z

        recon = z @ self.R.T
        resid = ((h - recon) ** 2).sum(dim=1, keepdim=True)
        info["resid_mean"] = resid.mean().detach()
        payload = torch.cat([z, resid], dim=1)
        return payload, info

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = payload[:, :self.k]  # strip residual scalar
        return z @ self.R.T


class FixedRandomMLPLiftBottleneck(nn.Module):
    """Fixed random projection (client) + learned MLP lift-back (server).

    Client compresses with a fixed orthogonal matrix R (client-only).
    Server learns a nonlinear MLP to map the k-dim payload back to
    activation_dim — it has NO access to R.
    """
    def __init__(self, activation_dim: int, k: int,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

        hidden = max(2 * k, 1024)
        self.learned_lift = nn.Sequential(
            nn.Linear(k, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, activation_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        info = {}
        if self.clip_norm > 0:
            norms = h.norm(dim=1, keepdim=True)
            clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
            h = h * clip_factor
            info["clip_fraction"] = (norms.squeeze() > self.clip_norm).float().mean().detach()
            info["activation_norm_mean"] = norms.mean().detach()
        z = h @ self.R
        if self.noise_multiplier > 0 and self.clip_norm > 0:
            noise_std = self.noise_multiplier * self.clip_norm
            z = z + torch.randn_like(z) * noise_std
            info["noise_std"] = noise_std
        info["bottleneck_z"] = z
        return z, info

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.learned_lift(payload)


class FixedRandomConvBottleneck(nn.Module):
    """Fixed random conv (client) + learned conv lift-back (server).
    Spatial structure preserved. Server never sees the random weights.
    kernel_size=1: channel-only mixing. kernel_size=3+: also mixes spatial neighbors."""
    def __init__(self, in_channels: int, k_channels: int, kernel_size: int = 1):
        super().__init__()
        self.k_channels = k_channels
        pad = kernel_size // 2  # same-padding to preserve spatial dims
        self.compress = nn.Conv2d(in_channels, k_channels, kernel_size,
                                  padding=pad, bias=False)
        with torch.no_grad():
            # Orthogonalize the center slice; rest stays random
            w = self.compress.weight.data
            c_out, c_in = w.shape[:2]
            flat = w.view(c_out, -1)  # (k_ch, in_ch * ks * ks)
            q, _ = torch.linalg.qr(flat.T)  # (in_ch*ks*ks, k_ch)
            self.compress.weight.data = q.T.view_as(w)
        self.compress.weight.requires_grad_(False)
        self.decompress = nn.Conv2d(k_channels, in_channels, kernel_size,
                                    padding=pad)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.compress(z_spatial), {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decompress(payload)


class Learned1x1Bottleneck(nn.Module):
    def __init__(self, in_channels: int, k_channels: int):
        super().__init__()
        self.k_channels = k_channels
        self.compress = nn.Conv2d(in_channels, k_channels, 1)
        self.decompress = nn.Conv2d(k_channels, in_channels, 1)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.compress(z_spatial), {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decompress(payload)


class SplitModel(nn.Module):
    def __init__(self, dataset: str, mode: str, k: int, deep_head: bool = False,
                 conv_kernel_size: int = 1,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0,
                 block_size: int = 2):
        super().__init__()
        cfg = MODEL_CONFIGS[dataset]
        self.activation_dim = cfg["activation_dim"]
        self.mode = mode
        self.k = k

        # Client: head + tail
        self.head = build_head(dataset, deep=deep_head)
        self.tail = build_tail(dataset)

        # Server: backbone
        self.backbone = build_backbone(dataset)

        # Bottleneck (between head and backbone)
        self.spatial_shape = cfg["spatial_shape"]  # (C, H, W)
        if mode == "raw":
            self.bottleneck = RawBottleneck()
        elif mode == "fixed_random":
            self.bottleneck = FixedRandomBottleneck(activation_dim=self.activation_dim, k=k,
                                                    clip_norm=clip_norm, noise_multiplier=noise_multiplier)
        elif mode == "fixed_random_resid":
            self.bottleneck = FixedRandomResidualBottleneck(activation_dim=self.activation_dim, k=k,
                                                            clip_norm=clip_norm, noise_multiplier=noise_multiplier)
        elif mode == "fixed_random_ll":
            self.bottleneck = FixedRandomMLPLiftBottleneck(activation_dim=self.activation_dim, k=k,
                                                            clip_norm=clip_norm, noise_multiplier=noise_multiplier)
        elif mode == "fixed_random_conv":
            in_ch = cfg["spatial_shape"][0]
            k_channels = in_ch // (self.activation_dim // k) if k < self.activation_dim else in_ch
            k_channels = max(k_channels, 1)
            self.bottleneck = FixedRandomConvBottleneck(in_channels=in_ch, k_channels=k_channels,
                                                         kernel_size=conv_kernel_size)
            self._k_channels = k_channels
        elif mode == "learned_1x1":
            in_ch = cfg["spatial_shape"][0]  # 64 for ResNet18
            k_channels = in_ch // (self.activation_dim // k) if k < self.activation_dim else in_ch
            k_channels = max(k_channels, 1)
            self.bottleneck = Learned1x1Bottleneck(in_channels=in_ch, k_channels=k_channels)
            self._k_channels = k_channels
        elif mode == "block_random":
            C, H, W = cfg["spatial_shape"]
            self.block_size = block_size
            self.bottleneck = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W,
                                                    block=block_size, k=k,
                                                    clip_norm=clip_norm,
                                                    noise_multiplier=noise_multiplier)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def payload_dim(self) -> int:
        if self.mode == "raw":
            return self.activation_dim
        if self.mode == "block_random":
            return self.bottleneck.payload_floats
        if self.mode in ("learned_1x1", "fixed_random_conv"):
            h, w = self.spatial_shape[1], self.spatial_shape[2]
            return self._k_channels * h * w
        if self.mode in ("fixed_random", "fixed_random_ll"):
            return self.k
        return self.k + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Client: head
        z_spatial = self.head(x)

        if isinstance(self.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
            # Learned 1x1: operates on spatial tensors directly
            compressed, info = self.bottleneck(z_spatial)
            z_approx = self.bottleneck.lift_back(compressed, info)
        else:
            # All other modes: flatten -> compress -> lift_back -> reshape
            z = z_spatial.flatten(1)
            payload, info = self.bottleneck(z)
            z_approx = self.bottleneck.lift_back(payload, info)
            z_approx = z_approx.reshape_as(z_spatial)

        # Reconstruction loss (computed during training if requested)
        if self.training and self.mode != "raw":
            info["recon_loss"] = F.mse_loss(z_approx, z_spatial.detach())

        backbone_out = self.backbone(z_approx)

        # Client: tail
        logits = self.tail(backbone_out)
        return logits, info


@dataclass
class EpochStats:
    loss: float
    acc: float
    samples: int
    comm_floats_per_sample: float


@dataclass
class RunSummary:
    dataset: str
    mode: str
    k: int
    activation_dim: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    best_global_test_acc: float
    final_global_test_acc: float
    final_train_acc: float
    payload_dim: int
    compression_ratio: float
    num_clients: int
    alpha: float
    deep_head: bool = False
    recon_lambda: float = 0.0
    clip_norm: float = 0.0
    noise_multiplier: float = 0.0
    wcc_lambda: float = 0.0


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return int((preds == y).sum().item())


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
    },
}


def get_datasets(data_root: str, dataset: str = "cifar10"):
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
    return train_set, test_set


def get_targets(dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise ValueError("Dataset does not expose .targets")


def dirichlet_split(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[Subset]:
    """Partition dataset across clients using a Dirichlet distribution.

    For each class, sample a proportion vector from Dir(alpha) and assign
    samples accordingly.  Higher alpha → more IID; lower alpha → more skewed.
    """
    targets = np.array(get_targets(dataset))
    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to actual counts (cumulative split)
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)
        splits = np.clip(splits, 0, len(class_idx))
        parts = np.split(class_idx, splits[:-1])

        for client_id, part in enumerate(parts):
            client_indices[client_id].extend(part.tolist())

    return [Subset(dataset, idxs) for idxs in client_indices]


def make_client_loaders(
    subsets: List[Subset],
    batch_size: int,
    num_workers: int,
    train: bool,
) -> List[DataLoader]:
    loaders = []
    for subset in subsets:
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=train,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=train,
            )
        )
    return loaders


def get_multiclient_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    num_clients: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
    dataset: str = "cifar10",
) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_set, test_set = get_datasets(data_root, dataset)

    train_subsets = dirichlet_split(train_set, num_clients, alpha, seed=seed)
    test_subsets = dirichlet_split(test_set, num_clients, alpha, seed=seed + 1)

    train_loaders = make_client_loaders(
        train_subsets, batch_size=batch_size, num_workers=num_workers, train=True
    )
    test_loaders = make_client_loaders(
        test_subsets, batch_size=batch_size, num_workers=num_workers, train=False
    )

    return train_loaders, test_loaders


def run_multiclient_epoch(
    model: SplitModel,
    client_loaders: List[DataLoader],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    recon_lambda: float = 0.0,
    wcc_lambda: float = 0.0,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_comm = 0.0

    for client_id, loader in enumerate(client_loaders):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.set_grad_enabled(is_train):
                logits, info = model(x)
                loss = F.cross_entropy(logits, y)
                if recon_lambda > 0 and "recon_loss" in info:
                    loss = loss + recon_lambda * info["recon_loss"]

                # Within-Class Compaction: minimize I(z; x | y)
                if wcc_lambda > 0 and "bottleneck_z" in info:
                    z_bn = info["bottleneck_z"]  # (B, k)
                    # Compute per-class centroids and compaction loss
                    wcc_loss = torch.tensor(0.0, device=device)
                    for c in y.unique():
                        mask = (y == c)
                        if mask.sum() > 1:
                            z_c = z_bn[mask]
                            mu_c = z_c.mean(dim=0, keepdim=True)
                            wcc_loss = wcc_loss + (z_c - mu_c).pow(2).sum(dim=1).mean()
                    loss = loss + wcc_lambda * wcc_loss

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_correct += accuracy_from_logits(logits, y)
            total_samples += bsz
            total_comm += model.payload_dim * bsz

    return EpochStats(
        loss=total_loss / total_samples,
        acc=total_correct / total_samples,
        samples=total_samples,
        comm_floats_per_sample=total_comm / total_samples,
    )


def eval_per_client(
    model: SplitModel,
    client_test_loaders: List[DataLoader],
    device: torch.device,
) -> Dict[int, float]:
    model.eval()
    out: Dict[int, float] = {}

    with torch.no_grad():
        for client_id, loader in enumerate(client_test_loaders):
            correct = 0
            total = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits, _ = model(x)
                correct += accuracy_from_logits(logits, y)
                total += y.size(0)

            out[client_id] = correct / total if total > 0 else 0.0

    return out


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_completed_runs(epoch_csv: str, target_epochs: int) -> set:
    """Return set of (alpha, num_clients, mode, k) tuples that already have target_epochs logged."""
    path = Path(epoch_csv)
    if not path.exists():
        return set()
    counts: Dict[Tuple[float, int, str, int], int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = float(row["alpha"]) if "alpha" in row else 1.0
            nc = int(row["num_clients"]) if "num_clients" in row else 5
            key = (alpha, nc, row["mode"], int(row["k"]))
            counts[key] = counts.get(key, 0) + 1
    return {k for k, v in counts.items() if v >= target_epochs}


def run_single_experiment(
    args,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    mode: str,
    k: int,
) -> RunSummary:
    dataset = getattr(args, "dataset", "cifar10")
    model = SplitModel(
        dataset=dataset,
        mode=mode,
        k=k,
        deep_head=getattr(args, "deep_head", False),
        conv_kernel_size=getattr(args, "conv_kernel_size", 1),
        clip_norm=getattr(args, "clip_norm", 0.0),
        noise_multiplier=getattr(args, "noise_multiplier", 0.0),
    ).to(device)

    activation_dim = model.activation_dim

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if not getattr(args, "no_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    payload_dim = model.payload_dim
    compression_ratio = activation_dim / payload_dim

    print("=" * 88)
    print(f"dataset            : {dataset}")
    print(f"mode               : {mode}")
    print(f"k                  : {k}")
    print(f"activation_dim (d) : {activation_dim}")
    print(f"device             : {device}")
    print(f"head params        : {count_parameters(model.head):,}")
    print(f"backbone params    : {count_parameters(model.backbone):,}")
    print(f"tail params        : {count_parameters(model.tail):,}")
    print(f"total params       : {count_parameters(model):,}")
    print(f"payload/sample     : {payload_dim} floats")
    print(f"compression ratio  : {compression_ratio:.2f}x")
    print(f"num clients        : {len(train_loaders)}")
    print(f"alpha (Dirichlet)  : {args.alpha}")
    for cid, loader in enumerate(train_loaders):
        print(f"client {cid} train samples : {len(loader.dataset)}")
    print("=" * 88)

    best_test = 0.0
    final_train = 0.0
    final_test = 0.0
    final_per_client: Dict[int, float] = {}
    epoch_history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        recon_lambda = getattr(args, "recon_lambda", 0.0)
        wcc_lambda = getattr(args, "wcc_lambda", 0.0)
        train_stats = run_multiclient_epoch(model, train_loaders, device, optimizer=optimizer, recon_lambda=recon_lambda, wcc_lambda=wcc_lambda)
        test_stats = run_multiclient_epoch(model, test_loaders, device, optimizer=None)
        per_client = eval_per_client(model, test_loaders, device)

        best_test = max(best_test, test_stats.acc)
        final_train = train_stats.acc
        final_test = test_stats.acc
        final_per_client = per_client

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_acc": train_stats.acc,
            "test_loss": test_stats.loss,
            "test_acc": test_stats.acc,
            "per_client_acc": dict(per_client),
        })

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | lr {current_lr:.6f} | "
            f"train loss {train_stats.loss:.4f} | train acc {train_stats.acc*100:6.2f}% | "
            f"global test loss {test_stats.loss:.4f} | global test acc {test_stats.acc*100:6.2f}% | "
            f"comm {test_stats.comm_floats_per_sample:.1f} floats/sample"
        )
        print(
            "per-client test acc | " +
            " | ".join([f"c{cid}: {acc*100:5.2f}%" for cid, acc in per_client.items()])
        )

        if scheduler is not None:
            scheduler.step()

        if args.epoch_csv is not None:
            row = {
                "dataset": getattr(args, "dataset", "cifar10"),
                "alpha": args.alpha,
                "num_clients": len(train_loaders),
                "mode": mode,
                "k": k,
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_acc": train_stats.acc,
                "global_test_loss": test_stats.loss,
                "global_test_acc": test_stats.acc,
                "payload_dim": payload_dim,
                "compression_ratio": compression_ratio,
                "deep_head": getattr(args, "deep_head", False),
                "recon_lambda": getattr(args, "recon_lambda", 0.0),
                "clip_norm": getattr(args, "clip_norm", 0.0),
                "noise_multiplier": getattr(args, "noise_multiplier", 0.0),
                "wcc_lambda": getattr(args, "wcc_lambda", 0.0),
            }
            for cid, acc in per_client.items():
                row[f"client_{cid}_test_acc"] = acc
            append_csv_row(Path(args.epoch_csv), row)

    summary = RunSummary(
        dataset=getattr(args, "dataset", "cifar10"),
        mode=mode,
        k=k,
        activation_dim=activation_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        best_global_test_acc=best_test,
        final_global_test_acc=final_test,
        final_train_acc=final_train,
        payload_dim=payload_dim,
        compression_ratio=compression_ratio,
        num_clients=len(train_loaders),
        alpha=args.alpha,
        deep_head=getattr(args, "deep_head", False),
        recon_lambda=getattr(args, "recon_lambda", 0.0),
        clip_norm=getattr(args, "clip_norm", 0.0),
        noise_multiplier=getattr(args, "noise_multiplier", 0.0),
        wcc_lambda=getattr(args, "wcc_lambda", 0.0),
    )

    print("-" * 88)
    print(f"best global test acc  : {best_test*100:.2f}%")
    print(f"final global test acc : {final_test*100:.2f}%")
    print(
        "final per-client acc  : " +
        " | ".join([f"c{cid}: {acc*100:5.2f}%" for cid, acc in final_per_client.items()])
    )
    print()

    if args.summary_csv is not None:
        row = asdict(summary)
        for cid, acc in final_per_client.items():
            row[f"client_{cid}_final_test_acc"] = acc
        append_csv_row(Path(args.summary_csv), row)


    return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")

    parser.add_argument("--mode", type=str,
                        choices=["raw", "fixed_random", "fixed_random_resid", "fixed_random_conv", "learned_1x1"],
                        default="fixed_random")
    parser.add_argument("--compression-ratio", type=int, default=16,
                        help="Compression ratio (d/k). k = activation_dim // compression_ratio")
    parser.add_argument("--conv-kernel-size", type=int, default=1,
                        help="Kernel size for fixed_random_conv compress/decompress (1=channel-only, 3=spatial mixing)")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-clients-list", type=int, nargs="+", default=None,
                        help="Sweep over multiple client counts (overrides --num-clients in sweep mode)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet concentration param (higher = more IID)")
    parser.add_argument("--alpha-list", type=float, nargs="+", default=None,
                        help="Sweep over multiple alpha values (e.g. --alpha-list 0.1 1e7 for non-IID/IID)")

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--modes", nargs="+",
                        default=["raw", "fixed_random", "fixed_random_resid", "fixed_random_conv", "learned_1x1"])
    parser.add_argument("--compression-ratio-list", nargs="+", type=int, default=[8, 16, 32],
                        help="Compression ratios to sweep (d/k)")

    parser.add_argument("--summary-csv", type=str, default="results/csv/finder_multiclient_summary.csv")
    parser.add_argument("--epoch-csv", type=str, default="results/csv/finder_multiclient_epochs.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (mode, k) pairs that already finished in epoch CSV")
    parser.add_argument("--deep-head", action="store_true",
                        help="Use deeper 2-block head (~46.5K params instead of ~9.5K)")
    # Exploratory; not reported in the paper.
    parser.add_argument("--recon-lambda", type=float, default=0.0,
                        help="Reconstruction loss weight (0 = off)")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Disable cosine annealing LR scheduler (default: scheduler ON)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure generation after each experiment")
    parser.add_argument("--fig-dir", type=str, default="./results/figures",
                        help="Base directory for visualization output")
    parser.add_argument("--clip-norm", type=float, default=0.0,
                        help="Per-sample L2 clip norm for activations (0 = disabled)")
    parser.add_argument("--noise-multiplier", type=float, default=0.0,
                        help="Gaussian noise multiplier sigma (noise_std = sigma * clip_norm)")
    parser.add_argument("--wcc-lambda", type=float, default=0.0,
                        help="Within-class compaction regularizer weight (0 = off). "
                             "Minimizes I(z; x | y), preserves I(z; y).")

    return parser.parse_args()


def _run_sweep_for_setting(args, alpha, num_clients, device, completed, results):
    """Run all (mode, k) combinations for a given (alpha, num_clients)."""
    print(f"\n{'#' * 88}")
    print(f"# alpha = {alpha}, num_clients = {num_clients}")
    print(f"{'#' * 88}")

    train_loaders, test_loaders = get_multiclient_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_clients=num_clients,
        alpha=alpha,
        seed=args.seed,
        dataset=args.dataset,
    )

    # temporarily override args for logging inside run_single_experiment
    orig_nc, orig_alpha = args.num_clients, args.alpha
    args.num_clients = num_clients
    args.alpha = alpha

    activation_dim = MODEL_CONFIGS[args.dataset]["activation_dim"]

    for mode in args.modes:
        if mode == "raw":
            if (alpha, num_clients, "raw", 0) in completed:
                print(f"  skipping a={alpha} nc={num_clients} mode=raw k=0 (already done)")
            else:
                results.append(
                    run_single_experiment(
                        args, train_loaders, test_loaders, device, mode="raw", k=0
                    )
                )
        else:
            for cr in args.compression_ratio_list:
                k = activation_dim // cr
                if (alpha, num_clients, mode, k) in completed:
                    print(f"  skipping a={alpha} nc={num_clients} mode={mode} k={k} (already done)")
                else:
                    results.append(
                        run_single_experiment(
                            args, train_loaders, test_loaders, device, mode=mode, k=k
                        )
                    )

    args.num_clients = orig_nc
    args.alpha = orig_alpha


def main():
    args = parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    completed = get_completed_runs(args.epoch_csv, args.epochs) if args.resume else set()
    if completed:
        print(f"Resuming: skipping {len(completed)} already-completed runs: {completed}")

    if args.sweep:
        nc_list = args.num_clients_list or [args.num_clients]
        alpha_list = args.alpha_list or [args.alpha]
        results: List[RunSummary] = []

        for alpha in alpha_list:
            for nc in nc_list:
                _run_sweep_for_setting(args, alpha, nc, device, completed, results)

        print("=" * 88)
        print("FINAL SWEEP SUMMARY (this run)")
        print("=" * 88)
        for r in results:
            print(
                f"a={r.alpha:<10.1f} | nc={r.num_clients:3d} | mode={r.mode:10s} | k={r.k:3d} | "
                f"best_global_test={r.best_global_test_acc*100:6.2f}% | "
                f"payload={r.payload_dim:3d} | compression={r.compression_ratio:5.2f}x"
            )
    else:
        train_loaders, test_loaders = get_multiclient_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
            dataset=args.dataset,
        )
        activation_dim = MODEL_CONFIGS[args.dataset]["activation_dim"]
        k = 0 if args.mode == "raw" else activation_dim // args.compression_ratio
        if (args.alpha, args.num_clients, args.mode, k) in completed:
            print(f"  skipping a={args.alpha} nc={args.num_clients} mode={args.mode} k={k} (already done)")
        else:
            run_single_experiment(
                args, train_loaders, test_loaders, device, mode=args.mode, k=k
            )


if __name__ == "__main__":
    main()