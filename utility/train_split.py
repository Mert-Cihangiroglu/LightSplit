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
import shutil
import urllib.request
import zipfile

from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from client_models import (build_head, build_backbone, build_tail, MODEL_CONFIGS,
                           cut_dims, native_cut_channels)


class RawBottleneck(nn.Module):
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return h, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload

# Honest transmitted-byte accounting for sparse payloads:
# each kept coordinate costs a 4-byte float32 value + a 2-byte uint16 index.
# (uint16 index covers activation_dim up to 65535; max d here is 16384.)
VALUE_BYTES = 4
INDEX_BYTES = 2


class TopKSparseBottleneck(nn.Module):
    """True magnitude-based top-k sparsification (matched-byte compression baseline).

    Keeps the k_s largest-magnitude coordinates per sample and zeroes the rest. The
    payload stays dense (d-dim) so the existing forward/reshape/recon path is reused
    unchanged; the honest transmitted cost is carried by SplitModel.payload_dim.
    Not compatible with WCC (it emits no projected z), which is correct: WCC is a
    LightSplit component, not part of this baseline. Gradients flow straight through
    the kept coordinates (mask-multiply), i.e. trained with the sparsifier in place.
    """

    def __init__(self, k_s: int):
        super().__init__()
        self.k_s = k_s

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, dim = h.shape
        k = min(self.k_s, dim)
        _, idx = torch.topk(h.abs(), k, dim=1)
        mask = torch.zeros_like(h).scatter(1, idx, 1.0)   # 0/1 keep-mask, per sample
        return h * mask, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload


class FixedRandomBottleneck(nn.Module):
    def __init__(self, activation_dim: int, k: int):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        return z, {"bottleneck_z": z}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload @ self.R.T


class FixedRandomResidualBottleneck(nn.Module):
    def __init__(self, activation_dim: int, k: int):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        recon = z @ self.R.T
        resid = ((h - recon) ** 2).sum(dim=1, keepdim=True)
        payload = torch.cat([z, resid], dim=1)
        return payload, {"resid_mean": resid.mean().detach()}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = payload[:, :self.k]  # strip residual scalar
        return z @ self.R.T


class FixedRandomMLPLiftBottleneck(nn.Module):
    """Fixed random projection (client) + learned MLP lift-back (server).

    Client compresses with a fixed orthogonal matrix R (client-only).
    Server learns a nonlinear MLP to map the k-dim payload back to
    activation_dim — it has NO access to R.
    """
    def __init__(self, activation_dim: int, k: int, hidden: int, store_z: bool = False):
        super().__init__()
        # R is client-side only — server never sees it
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k
        self.hidden = hidden
        self.store_z = store_z

        # Server-side MLP: learns to interpret compressed payload
        self.learned_lift = nn.Sequential(
            nn.Linear(k, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, activation_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        info = {}
        if self.store_z:
            info["bottleneck_z"] = z
        return z, info

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.learned_lift(payload)


class DeepConvDecompress(nn.Module):
    """Multi-layer CNN decoder for inverting spatial conv compression.
    First layer uses the same kernel as compress (unmix at same receptive field),
    second layer adds nonlinear refinement, final 1x1 projects to output channels."""
    def __init__(self, k_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        mid = max(2 * k_channels, 32)
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(k_channels, mid, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedRandomConvBottleneck(nn.Module):
    """Fixed random conv (client) + learned conv lift-back (server).
    Spatial structure preserved. Server never sees the random weights.
    kernel_size=1: channel-only mixing. kernel_size=3+: also mixes spatial neighbors.

    backbone_adapter: if True, use a simple 1x1 Conv+BN adapter instead of
        trying to invert the random conv.  The adapter learns to map the
        compressed k-channel representation to something the backbone can
        consume — it does NOT attempt reconstruction."""
    def __init__(self, in_channels: int, k_channels: int, kernel_size: int = 1,
                 deep_decompress: bool = False, backbone_adapter: bool = False):
        super().__init__()
        self.k_channels = k_channels
        self.backbone_adapter = backbone_adapter
        pad = kernel_size // 2
        self.compress = nn.Conv2d(in_channels, k_channels, kernel_size,
                                  padding=pad, bias=False)
        with torch.no_grad():
            w = self.compress.weight.data
            c_out, c_in = w.shape[:2]
            flat = w.view(c_out, -1)
            q, _ = torch.linalg.qr(flat.T)
            self.compress.weight.data = q.T.view_as(w)
        self.compress.weight.requires_grad_(False)
        if backbone_adapter:
            # Adapter: linear conv (same kernel as compress) + BN. No ReLU.
            # Not trying to invert — just giving backbone learnable features.
            self.decompress = nn.Sequential(
                nn.Conv2d(k_channels, in_channels, kernel_size, padding=pad, bias=False),
                nn.BatchNorm2d(in_channels),
            )
        elif deep_decompress and kernel_size > 1:
            self.decompress = DeepConvDecompress(k_channels, in_channels, kernel_size)
        else:
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


class BlockRandomBottleneck(nn.Module):
    """Block-local fixed orthonormal random projection (privacy/utility dial).

    Tiles the (C,H,W) cut tensor into block x block spatial tiles, flattens each
    tile to C*block*block dims, and projects it with a fixed orthonormal random R
    to out_per_block dims.  The macro grid (H/block x W/block) is preserved, so the
    random mixing is confined to a block x block receptive field -- the knob between
    privacy and utility:
        block = 1   -> per-position channel projection (max utility, min privacy)
        block = H   -> single global dense R == FixedRandomBottleneck (max privacy)
    Operates on the spatial tensor (like Learned1x1 / FixedRandomConv).  One R is
    shared across tiles and kept as a client-only buffer (never trained, never sent).
    Transmitted payload = n_blocks * out_per_block floats (matched-byte ~= k).
    """
    def __init__(self, channels: int, hw_h: int, hw_w: int, block: int, k: int):
        super().__init__()
        if hw_h % block != 0 or hw_w % block != 0:
            raise ValueError(f"block={block} must divide spatial dims ({hw_h},{hw_w})")
        self.channels, self.h, self.w, self.block = channels, hw_h, hw_w, block
        self.grid_h, self.grid_w = hw_h // block, hw_w // block
        self.n_blocks = self.grid_h * self.grid_w
        self.in_per_block = channels * block * block
        # floor-divide the float budget across tiles (exact for square 2^n grids; the
        # payload_dim property reports the honest transmitted size for other shapes).
        self.out_per_block = max(1, k // self.n_blocks)
        self.payload_floats = self.n_blocks * self.out_per_block
        rand_matrix = torch.randn(self.in_per_block, self.out_per_block)
        q, _ = torch.linalg.qr(rand_matrix)        # (in_per_block, out_per_block), orthonormal cols
        self.register_buffer("R", q)               # shared across tiles; client-only

    def _to_blocks(self, z: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, n_blocks, in_per_block)
        b = z.shape[0]
        x = z.unfold(2, self.block, self.block).unfold(3, self.block, self.block)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()          # (B, grid_h, grid_w, C, block, block)
        return x.reshape(b, self.n_blocks, self.in_per_block)

    def _from_blocks(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_blocks, in_per_block) -> (B, C, H, W)
        b = x.shape[0]
        x = x.reshape(b, self.grid_h, self.grid_w, self.channels, self.block, self.block)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()          # (B, C, grid_h, block, grid_w, block)
        return x.reshape(b, self.channels, self.h, self.w)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        payload = self._to_blocks(z_spatial) @ self.R          # (B, n_blocks, out_per_block)
        return payload, {"bottleneck_z": payload.flatten(1)}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._from_blocks(payload @ self.R.transpose(0, 1))


class BlockRandomMLPLiftBottleneck(nn.Module):
    """Block-local fixed random projection (client) + learned MLP lift-back (server).

    Same client compression as BlockRandomBottleneck -- tiles (C,H,W) into block x
    block spatial tiles and projects each with one shared fixed orthonormal R (a
    client-only buffer, never trained, never sent).  But the server does NOT invert
    with R^T; instead it learns an MLP that maps the k-dim payload back to the full
    activation_dim (it never sees R), then reshapes to (C,H,W) for the backbone.
    This is the block analogue of FixedRandomMLPLiftBottleneck -- the "MLP-liftback"
    server mode -- vs BlockRandomBottleneck's transpose->backbone server mode.
    """
    def __init__(self, channels: int, hw_h: int, hw_w: int, block: int, k: int,
                 hidden: int, store_z: bool = False):
        super().__init__()
        if hw_h % block != 0 or hw_w % block != 0:
            raise ValueError(f"block={block} must divide spatial dims ({hw_h},{hw_w})")
        self.channels, self.h, self.w, self.block = channels, hw_h, hw_w, block
        self.grid_h, self.grid_w = hw_h // block, hw_w // block
        self.n_blocks = self.grid_h * self.grid_w
        self.in_per_block = channels * block * block
        self.out_per_block = max(1, k // self.n_blocks)
        self.payload_floats = self.n_blocks * self.out_per_block
        self.activation_dim = channels * hw_h * hw_w
        self.store_z = store_z
        rand_matrix = torch.randn(self.in_per_block, self.out_per_block)
        q, _ = torch.linalg.qr(rand_matrix)        # orthonormal cols; shared across tiles, client-only
        self.register_buffer("R", q)
        # Server-side MLP: lifts the k-dim payload back to activation_dim (no access to R).
        self.learned_lift = nn.Sequential(
            nn.Linear(self.payload_floats, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.activation_dim),
        )

    def _to_blocks(self, z: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, n_blocks, in_per_block)
        b = z.shape[0]
        x = z.unfold(2, self.block, self.block).unfold(3, self.block, self.block)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()          # (B, grid_h, grid_w, C, block, block)
        return x.reshape(b, self.n_blocks, self.in_per_block)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        payload = self._to_blocks(z_spatial) @ self.R          # (B, n_blocks, out_per_block)
        info = {}
        if self.store_z:
            info["bottleneck_z"] = payload.flatten(1)
        return payload, info

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        lifted = self.learned_lift(payload.flatten(1))         # (B, activation_dim)
        return lifted.reshape(-1, self.channels, self.h, self.w)


class SplitModel(nn.Module):
    def __init__(self, dataset: str, mode: str, k: int, deep_head: bool = False,
                 num_clients: int = 1, client_heads: bool = False,
                 conv_kernel_size: int = 1, deep_decompress: bool = False,
                 backbone_adapter: bool = False, attn_layer1: bool = False,
                 backbone: str = "resnet18", wcc_lambda: float = 0.0,
                 mlp_hidden: int = None, block_size: int = 2,
                 center_lambda: float = 0.0,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0,
                 cut_channels: int = None, head_fixed_body: bool = False):
        super().__init__()
        cfg = MODEL_CONFIGS[dataset]
        # Cut-layer width the head emits cut_channels x H x W instead of the
        # native C x H x W.  Everything downstream (bottleneck dims, payload, CR) must
        # follow the ACTUAL cut, never the static config -- so derive both here and use
        # self.spatial_shape / self.activation_dim below, not cfg[...].
        C, H, W, act_dim = cut_dims(dataset, cut_channels)
        self.cut_channels = C
        self.spatial_shape = (C, H, W)
        self.activation_dim = act_dim
        self.mode = mode
        self.k = k
        self.client_heads = client_heads
        # DP on the transmitted payload: per-sample L2 clip (bounds sensitivity) then
        # calibrated Gaussian noise (std = noise_multiplier * clip_norm). Applied train
        # AND eval so the server truly receives a noised representation (real defense).
        # Ported from the reference SDAR implementation so DP baselines live in the utility harness like the rest.
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

        # Center loss (Wen et al. 2016): persistent per-class centers in bottleneck-z
        # space. For fixed_random, bottleneck_z is k-dim, so centers are [num_classes, k].
        # Updated by an EMA rule (not the optimizer); see run_multiclient_epoch.
        self.center_lambda = center_lambda
        if center_lambda > 0:
            n_cls = DATASET_STATS[dataset]["num_classes"]
            self.register_buffer("centers", torch.zeros(n_cls, k))

        # Client: head (per-client or shared)
        if client_heads:
            self.heads = nn.ModuleList([
                build_head(dataset, deep=deep_head, cut_channels=C, fixed_body=head_fixed_body)
                for _ in range(num_clients)
            ])
            self.head = None
        else:
            self.head = build_head(dataset, deep=deep_head, cut_channels=C, fixed_body=head_fixed_body)
            self.heads = None

        # Client: tail
        self.tail = build_tail(dataset)

        # Server: backbone (accepts the narrowed C-channel cut; widens it internally)
        self.backbone = build_backbone(dataset, backbone=backbone,
                                       attn_layer1=attn_layer1, cut_channels=C)

        # Bottleneck (between head and backbone)
        if mode == "raw":
            self.bottleneck = RawBottleneck()
        elif mode == "topk_sparse":
            # matched-byte: keep k_s coords so k_s*(4+2) bytes ~= k*4 bytes (LightSplit budget)
            self._k_s = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))
            self.bottleneck = TopKSparseBottleneck(k_s=self._k_s)
        elif mode == "fixed_random":
            self.bottleneck = FixedRandomBottleneck(activation_dim=self.activation_dim, k=k)
        elif mode == "fixed_random_resid":
            self.bottleneck = FixedRandomResidualBottleneck(activation_dim=self.activation_dim, k=k)
        elif mode == "fixed_random_ll":
            # Resolve MLP hidden dim: explicit arg > dataset config > back-compat fallback.
            if mlp_hidden is None:
                mlp_hidden = cfg.get("mlp_hidden", max(2 * k, 1024))
            self.mlp_hidden = mlp_hidden
            self.bottleneck = FixedRandomMLPLiftBottleneck(
                activation_dim=self.activation_dim, k=k,
                hidden=mlp_hidden,
                store_z=(wcc_lambda > 0))
        elif mode == "fixed_random_conv":
            in_ch = C  # cut channels (64 for native ResNet18)
            k_channels = in_ch // (self.activation_dim // k) if k < self.activation_dim else in_ch
            k_channels = max(k_channels, 1)
            self.bottleneck = FixedRandomConvBottleneck(in_channels=in_ch, k_channels=k_channels,
                                                         kernel_size=conv_kernel_size,
                                                         deep_decompress=deep_decompress,
                                                         backbone_adapter=backbone_adapter)
            self._k_channels = k_channels
        elif mode == "learned_1x1":
            in_ch = C  # cut channels (64 for native ResNet18)
            k_channels = in_ch // (self.activation_dim // k) if k < self.activation_dim else in_ch
            k_channels = max(k_channels, 1)
            self.bottleneck = Learned1x1Bottleneck(in_channels=in_ch, k_channels=k_channels)
            self._k_channels = k_channels
        elif mode == "block_random":
            self.block_size = block_size
            self.bottleneck = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W,
                                                    block=block_size, k=k)
        elif mode == "block_random_ll":
            self.block_size = block_size
            if mlp_hidden is None:
                mlp_hidden = cfg.get("mlp_hidden", max(2 * k, 1024))
            self.mlp_hidden = mlp_hidden
            self.bottleneck = BlockRandomMLPLiftBottleneck(
                channels=C, hw_h=H, hw_w=W, block=block_size, k=k,
                hidden=mlp_hidden, store_z=(wcc_lambda > 0))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def payload_dim(self) -> int:
        if self.mode == "raw":
            return self.activation_dim
        if self.mode in ("block_random", "block_random_ll"):
            return self.bottleneck.payload_floats
        if self.mode in ("learned_1x1", "fixed_random_conv"):
            h, w = self.spatial_shape[1], self.spatial_shape[2]
            return self._k_channels * h * w
        if self.mode == "topk_sparse":
            # honest byte cost expressed as float32-equivalents (ceil(6*k_s/4))
            return (self._k_s * (VALUE_BYTES + INDEX_BYTES) + VALUE_BYTES - 1) // VALUE_BYTES
        if self.mode in ("fixed_random", "fixed_random_ll"):
            return self.k
        return self.k + 1

    def _apply_dp(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample L2 clip + Gaussian noise on the transmitted payload (train+eval)."""
        if self.clip_norm <= 0:
            return payload
        flat = payload.flatten(1)
        norms = flat.norm(dim=1, keepdim=True)
        clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
        flat = flat * clip_factor
        info["dp_clip_fraction"] = (norms.squeeze() > self.clip_norm).float().mean().detach()
        if self.noise_multiplier > 0:
            flat = flat + torch.randn_like(flat) * (self.noise_multiplier * self.clip_norm)
        return flat.reshape_as(payload)

    def forward(self, x: torch.Tensor, client_id: int = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Client: head
        if self.client_heads:
            assert client_id is not None, "client_id required when using client-specific heads"
            z_spatial = self.heads[client_id](x)
        else:
            z_spatial = self.head(x)

        if isinstance(self.bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck, BlockRandomBottleneck, BlockRandomMLPLiftBottleneck)):
            # Learned 1x1 / block_random(+ll): operate on spatial tensors directly
            compressed, info = self.bottleneck(z_spatial)
            compressed = self._apply_dp(compressed, info)
            z_approx = self.bottleneck.lift_back(compressed, info)
        else:
            # All other modes: flatten -> compress -> lift_back -> reshape
            z = z_spatial.flatten(1)
            payload, info = self.bottleneck(z)
            payload = self._apply_dp(payload, info)
            z_approx = self.bottleneck.lift_back(payload, info)
            z_approx = z_approx.reshape_as(z_spatial)

        # Reconstruction loss (computed during training if requested)
        if self.training and self.mode != "raw":
            info["recon_loss"] = F.mse_loss(z_approx, z_spatial)

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
    wcc_lambda: float = 0.0
    center_lambda: float = 0.0
    center_alpha: float = 0.5
    clip_norm: float = 0.0
    noise_multiplier: float = 0.0
    client_heads: bool = False
    conv_kernel_size: int = 1
    deep_decompress: bool = False
    backbone_adapter: bool = False
    detach_bottleneck: bool = False
    attn_layer1: bool = False
    backbone: str = "resnet18"
    mlp_hidden: int = 0  # 0 = not applicable (non-FR-LL modes)
    cut_channels: int = 0  # head output channels (native when not narrowed)
    cut_width: int = 0     # achieved cut dim d = cut_channels * H * W
    head_fixed_body: bool = False  # cut-width sweep: native-width head body, only the cut stage varies


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
    "gtsrb": {
        "mean": (0.3403, 0.3121, 0.3214),
        "std":  (0.2724, 0.2608, 0.2669),
        "num_classes": 43,
    },
    "fmnist": {
        "mean": (0.2861,),
        "std":  (0.3530,),
        "num_classes": 10,
    },
}




def get_datasets(data_root: str, dataset: str = "cifar10"):
    stats = DATASET_STATS[dataset]

    if dataset in ("cifar10", "cifar100"):
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

    elif dataset == "gtsrb":
        train_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        test_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        train_set = datasets.GTSRB(root=data_root, split="train", download=True, transform=train_tf)
        test_set = datasets.GTSRB(root=data_root, split="test", download=True, transform=test_tf)

    elif dataset == "fmnist":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        train_set = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=test_tf)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_set, test_set


def get_targets(dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "_labels"):
        return list(dataset._labels)
    return [s[1] for s in dataset]


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
    center_lambda: float = 0.0,
    center_alpha: float = 0.5,
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
                logits, info = model(x, client_id=client_id)
                loss = F.cross_entropy(logits, y)
                if recon_lambda > 0 and "recon_loss" in info:
                    loss = loss + recon_lambda * info["recon_loss"]

                # Within-Class Compaction: minimize I(z; x | y)
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

                # Center loss (Wen et al. 2016): pull z toward persistent class centers.
                # IDENTICAL reduction to the WCC block above (sum over present classes of
                # the per-class mean squared distance) so the two are directly comparable
                # at matched lambda; the ONLY difference is that the target is a persistent,
                # EMA-updated center (carries state across batches) instead of the per-batch
                # class mean. Persistent centers give a valid target even for singleton
                # classes, so (unlike WCC) we do not skip them. Gradient flows only to the
                # features; centers are a detached buffer, EMA-updated after optimizer.step().
                if center_lambda > 0 and "bottleneck_z" in info:
                    z_bn = info["bottleneck_z"]
                    center_loss = torch.tensor(0.0, device=device)
                    for c in y.unique():
                        z_c = z_bn[y == c]
                        center_loss = center_loss + (z_c - model.centers[c]).pow(2).sum(dim=1).mean()
                    loss = loss + center_lambda * center_loss

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # EMA center update (Wen et al. 2016 rule), batch-present classes only.
                    if center_lambda > 0 and "bottleneck_z" in info:
                        with torch.no_grad():
                            z_det = info["bottleneck_z"].detach()
                            for c in y.unique():
                                m = (y == c)
                                model.centers[c] += center_alpha * (z_det[m].mean(dim=0) - model.centers[c])

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

                logits, _ = model(x, client_id=client_id)
                correct += accuracy_from_logits(logits, y)
                total += y.size(0)

            out[client_id] = correct / total if total > 0 else 0.0

    return out


def append_csv_row(path: Path, row: dict) -> None:
    """Append a row, honouring the file's EXISTING header if it has one.

    A new column in RunSummary must never shift the columns of a CSV written by an
    older schema: DictWriter takes its fieldnames from whatever header the file already
    declares, filling absent keys with "" and dropping keys the header doesn't know.
    Fresh files get a header from the row's own keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as f:
            header = next(csv.reader(f), None)
        if header:
            fieldnames = header
    with path.open("a", newline="") as f:
        if fieldnames is None:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
        else:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    restval="", extrasaction="ignore")
        writer.writerow(row)


def get_completed_runs(epoch_csv: str, target_epochs: int) -> set:
    """Return set of (alpha, num_clients, mode, k, client_heads, conv_kernel_size, block_size) tuples that already have target_epochs logged."""
    path = Path(epoch_csv)
    if not path.exists():
        return set()
    counts: Dict[Tuple, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = float(row["alpha"]) if "alpha" in row else 1.0
            nc = int(row["num_clients"]) if "num_clients" in row else 5
            ch = row.get("client_heads", "False") == "True"
            def _int_or(v, default):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return default
            ks = _int_or(row.get("conv_kernel_size", 1), 1)
            bs = _int_or(row.get("block_size", 1), 1)
            key = (alpha, nc, row["mode"], int(row["k"]), ch, ks, bs)
            counts[key] = counts.get(key, 0) + 1
    return {k for k, v in counts.items() if v >= target_epochs}


class _DualOptimizer:
    """Wraps two optimizers (client/server) behind a single interface."""
    def __init__(self, opt_client, opt_server):
        self.opt_client = opt_client
        self.opt_server = opt_server
        self.param_groups = opt_client.param_groups + opt_server.param_groups

    def zero_grad(self, set_to_none=False):
        self.opt_client.zero_grad(set_to_none=set_to_none)
        self.opt_server.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt_client.step()
        self.opt_server.step()


def run_single_experiment(
    args,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    mode: str,
    k: int,
) -> RunSummary:
    dataset = getattr(args, "dataset", "cifar10")
    if getattr(args, "backbone", "resnet18") == "vit" and getattr(args, "attn_layer1", False):
        print("WARNING: --attn-layer1 is ignored when --backbone=vit")
    model = SplitModel(
        dataset=dataset,
        mode=mode,
        k=k,
        deep_head=getattr(args, "deep_head", False),
        num_clients=len(train_loaders),
        client_heads=getattr(args, "client_heads", False),
        conv_kernel_size=getattr(args, "conv_kernel_size", 1),
        deep_decompress=getattr(args, "deep_decompress", False),
        backbone_adapter=getattr(args, "backbone_adapter", False),
        attn_layer1=getattr(args, "attn_layer1", False),
        backbone=getattr(args, "backbone", "resnet18"),
        wcc_lambda=getattr(args, "wcc_lambda", 0.0),
        mlp_hidden=getattr(args, "mlp_hidden", None),
        block_size=getattr(args, "block_size", 2),
        center_lambda=getattr(args, "center_lambda", 0.0),
        clip_norm=getattr(args, "clip_norm", 0.0),
        noise_multiplier=getattr(args, "noise_multiplier", 0.0),
        cut_channels=getattr(args, "cut_channels", None),
        head_fixed_body=getattr(args, "head_fixed_body", False),
    ).to(device)

    activation_dim = model.activation_dim

    decompress_lr = getattr(args, "decompress_lr", None) or args.lr
    detach_mode = getattr(args, "detach_bottleneck", False)

    if detach_mode:
        # SDAR-style: separate client/server optimizers, constant LR
        client_params = list(model.tail.parameters())
        if model.client_heads:
            client_params.extend(model.heads.parameters())
        else:
            client_params.extend(model.head.parameters())
        client_params = [p for p in client_params if p.requires_grad]

        server_params = list(model.backbone.parameters())
        if hasattr(model.bottleneck, 'decompress'):
            server_params.extend(model.bottleneck.decompress.parameters())
        server_params = [p for p in server_params if p.requires_grad]

        opt_client = torch.optim.Adam(client_params, lr=args.lr, weight_decay=args.weight_decay)
        opt_server = torch.optim.Adam(server_params, lr=decompress_lr, weight_decay=args.weight_decay)
        optimizer = _DualOptimizer(opt_client, opt_server)
        scheduler = None  # forced constant LR
    else:
        decompress_params = set()
        if hasattr(model.bottleneck, 'decompress'):
            decompress_params = set(model.bottleneck.decompress.parameters())
        base_params = [p for p in model.parameters() if p.requires_grad and p not in decompress_params]
        dec_params = [p for p in decompress_params if p.requires_grad]
        param_groups = [{"params": base_params, "lr": args.lr}]
        if dec_params:
            param_groups.append({"params": dec_params, "lr": decompress_lr})
        optimizer = torch.optim.Adam(
            param_groups,
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
    print(f"cut_channels       : {model.cut_channels}"
          f"{'' if model.cut_channels == native_cut_channels(dataset) else f' (NARROWED from {native_cut_channels(dataset)})'}")
    print(f"backbone           : {getattr(args, 'backbone', 'resnet18')}")
    print(f"device             : {device}")
    if model.client_heads:
        head_each = count_parameters(model.heads[0])
        head_total = count_parameters(model.heads)
        print(f"head params (x{len(model.heads)})  : {head_each:,} each, {head_total:,} total")
    else:
        print(f"head params        : {count_parameters(model.head):,}")
    print(f"backbone params    : {count_parameters(model.backbone):,}")
    print(f"tail params        : {count_parameters(model.tail):,}")
    if hasattr(model.bottleneck, 'decompress'):
        print(f"decompress params  : {count_parameters(model.bottleneck.decompress):,}")
    print(f"total params       : {count_parameters(model):,}")
    print(f"payload/sample     : {payload_dim} floats")
    if mode == "fixed_random_conv":
        print(f"conv_kernel_size   : {getattr(args, 'conv_kernel_size', 1)}")
        print(f"deep_decompress    : {getattr(args, 'deep_decompress', False)}")
        print(f"backbone_adapter   : {getattr(args, 'backbone_adapter', False)}")
        print(f"detach_bottleneck  : {detach_mode}")
        print(f"attn_layer1        : {getattr(args, 'attn_layer1', False)}")
        print(f"decompress_lr      : {decompress_lr}")
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
    patience = getattr(args, "patience", 0)
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        recon_lambda = getattr(args, "recon_lambda", 0.0)
        wcc_lambda = getattr(args, "wcc_lambda", 0.0)
        center_lambda = getattr(args, "center_lambda", 0.0)
        center_alpha = getattr(args, "center_alpha", 0.5)
        train_stats = run_multiclient_epoch(model, train_loaders, device, optimizer=optimizer, recon_lambda=recon_lambda, wcc_lambda=wcc_lambda, center_lambda=center_lambda, center_alpha=center_alpha)
        test_stats = run_multiclient_epoch(model, test_loaders, device, optimizer=None)
        per_client = eval_per_client(model, test_loaders, device)

        if test_stats.acc > best_test:
            best_test = test_stats.acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
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
                "cut_channels": model.cut_channels,
                "cut_width": model.activation_dim,
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_acc": train_stats.acc,
                "global_test_loss": test_stats.loss,
                "global_test_acc": test_stats.acc,
                "payload_dim": payload_dim,
                "compression_ratio": compression_ratio,
                "deep_head": getattr(args, "deep_head", False),
                "recon_lambda": getattr(args, "recon_lambda", 0.0),
                "wcc_lambda": getattr(args, "wcc_lambda", 0.0),
                "center_lambda": getattr(args, "center_lambda", 0.0),
                "client_heads": getattr(args, "client_heads", False),
                "conv_kernel_size": getattr(args, "conv_kernel_size", 1),
                "block_size": (getattr(args, "block_size", 1) if mode == "block_random" else 1),
                "deep_decompress": getattr(args, "deep_decompress", False),
                "backbone_adapter": getattr(args, "backbone_adapter", False),
                "detach_bottleneck": getattr(args, "detach_bottleneck", False),
                "attn_layer1": getattr(args, "attn_layer1", False),
                "backbone": getattr(args, "backbone", "resnet18"),
            }
            for cid, acc in per_client.items():
                row[f"client_{cid}_test_acc"] = acc
            append_csv_row(Path(args.epoch_csv), row)

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

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
        wcc_lambda=getattr(args, "wcc_lambda", 0.0),
        center_lambda=getattr(args, "center_lambda", 0.0),
        center_alpha=getattr(args, "center_alpha", 0.5),
        clip_norm=getattr(args, "clip_norm", 0.0),
        noise_multiplier=getattr(args, "noise_multiplier", 0.0),
        client_heads=getattr(args, "client_heads", False),
        conv_kernel_size=getattr(args, "conv_kernel_size", 1),
        deep_decompress=getattr(args, "deep_decompress", False),
        backbone_adapter=getattr(args, "backbone_adapter", False),
        detach_bottleneck=getattr(args, "detach_bottleneck", False),
        attn_layer1=getattr(args, "attn_layer1", False),
        backbone=getattr(args, "backbone", "resnet18"),
        mlp_hidden=(getattr(model, "mlp_hidden", 0) if mode == "fixed_random_ll" else 0),
        cut_channels=model.cut_channels,
        cut_width=model.activation_dim,
        head_fixed_body=getattr(args, "head_fixed_body", False),
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
                        choices=["cifar10", "cifar100", "gtsrb", "fmnist"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")

    # fixed_random_resid and fixed_random_conv are exploratory modes, not reported in the paper.
    parser.add_argument("--mode", type=str,
                        choices=["raw", "topk_sparse", "fixed_random", "fixed_random_resid", "fixed_random_ll", "fixed_random_conv", "learned_1x1", "block_random", "block_random_ll"],
                        default="fixed_random")
    parser.add_argument("--compression-ratio", type=int, default=16,
                        help="Compression ratio (d/k). k = activation_dim // compression_ratio")
    # Exploratory; not reported in the paper.
    parser.add_argument("--conv-kernel-size", type=int, default=1,
                        help="Kernel size for fixed_random_conv compress/decompress (1=channel-only, 3=spatial mixing)")
    parser.add_argument("--block-size", type=int, default=2,
                        help="block_random tile size: 1=per-position (max utility), cut_hw=global dense==fixed_random (max privacy)")
    parser.add_argument("--head-fixed-body", action="store_true",
                        help="With --deep-head + --cut-channels: keep the head BODY at the native "
                             "width (64 for cifar) and let only the final stage set the cut width, "
                             "mirroring the SDAR privacy harness's [16,32,C] width family. Without "
                             "this the whole head runs at width C, which confounds a narrow cut with "
                             "a weak client head. No-op at the native cut width.")
    parser.add_argument("--cut-channels", type=int, default=None,
                        help="Cut-layer width in CHANNELS . Default None = dataset-native "
                             "(cifar 64, gtsrb 29, fmnist 20). The head emits cut_channels x H x W, "
                             "so activation_dim = cut_channels*H*W and k = activation_dim // CR are "
                             "relative to the NARROWED cut. Orthogonal to --compression-ratio.")
    # Exploratory; not reported in the paper.
    parser.add_argument("--conv-kernel-size-list", type=int, nargs="+", default=None,
                        help="Sweep over kernel sizes for fixed_random_conv (e.g. --conv-kernel-size-list 1 3 5 9)")
    # Exploratory; not reported in the paper.
    parser.add_argument("--deep-decompress", action="store_true",
                        help="Use multi-layer CNN decoder for conv bottleneck (recommended for kernel>1)")
    # Exploratory; not reported in the paper.
    parser.add_argument("--backbone-adapter", action="store_true",
                        help="Use 1x1 Conv+BN adapter instead of trying to invert the random conv (no reconstruction attempt)")
    # Exploratory; not reported in the paper.
    parser.add_argument("--detach-bottleneck", action="store_true",
                        help="SDAR-style: separate client/server optimizers, constant LR")
    # Exploratory; not reported in the paper.
    parser.add_argument("--attn-layer1", action="store_true",
                        help="Replace backbone layer1 (3x3 convs) with self-attention blocks (spatial-agnostic)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "vit"],
                        help="Backbone architecture (resnet18 or vit)")
    parser.add_argument("--decompress-lr", type=float, default=None,
                        help="Learning rate for decompress parameters (default: same as --lr)")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-clients-list", type=int, nargs="+", default=None,
                        help="Sweep over multiple client counts (overrides --num-clients in sweep mode)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet concentration param (higher = more IID)")
    parser.add_argument("--alpha-list", type=float, nargs="+", default=None,
                        help="Sweep over multiple alpha values (e.g. --alpha-list 0.1 1e7 for non-IID/IID)")

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--modes", nargs="+",
                        default=["raw", "fixed_random", "fixed_random_resid", "fixed_random_ll", "fixed_random_conv", "learned_1x1"])
    parser.add_argument("--compression-ratio-list", nargs="+", type=int, default=[8, 16, 32],
                        help="Compression ratios to sweep (d/k)")

    parser.add_argument("--summary-csv", type=str, default="results/csv/finder_multiclient_summary.csv")
    parser.add_argument("--epoch-csv", type=str, default="results/csv/finder_multiclient_epochs.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (mode, k) pairs that already finished in epoch CSV")
    parser.add_argument("--deep-head", action="store_true",
                        help="Use deeper 2-block head (~46.5K params instead of ~9.5K)")
    parser.add_argument("--client-heads", action="store_true",
                        help="Give each client its own head (default: shared head)")
    # Exploratory; not reported in the paper.
    parser.add_argument("--recon-lambda", type=float, default=0.0,
                        help="Reconstruction loss weight (0 = off)")
    parser.add_argument("--wcc-lambda", type=float, default=0.0,
                        help="Within-class compaction regularizer weight (0 = off). "
                             "Minimizes I(z; x | y), preserves I(z; y).")
    parser.add_argument("--center-lambda", type=float, default=0.0,
                        help="Center loss (Wen et al. 2016) weight (0 = off). Same class-"
                             "compaction objective as WCC but with persistent EMA centers "
                             "instead of per-batch means. Set EITHER --wcc-lambda or this.")
    parser.add_argument("--center-alpha", type=float, default=0.5,
                        help="EMA update rate for center-loss class centers (Wen et al. 2016).")
    parser.add_argument("--clip-norm", type=float, default=0.0,
                        help="DP: per-sample L2 clip norm on the transmitted payload (0 = off).")
    parser.add_argument("--noise-multiplier", type=float, default=0.0,
                        help="DP: Gaussian noise multiplier; std = noise_multiplier * clip_norm "
                             "(requires --clip-norm > 0).")
    parser.add_argument("--mlp-hidden", type=int, default=None,
                        help="MLP hidden dim for fixed_random_ll mode. "
                             "Default: MODEL_CONFIGS[dataset]['mlp_hidden'] if set, "
                             "else max(2*k, 1024) for backward compatibility.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled, run all epochs)")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Disable cosine annealing LR scheduler (default: scheduler ON)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure generation after each experiment")
    parser.add_argument("--fig-dir", type=str, default="./results/figures",
                        help="Base directory for visualization output")

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

    # k is relative to the ACTUAL cut width, which --cut-channels may have narrowed.
    activation_dim = cut_dims(args.dataset, getattr(args, "cut_channels", None))[3]
    ch = getattr(args, "client_heads", False)

    ks_default = getattr(args, "conv_kernel_size", 1)

    for mode in args.modes:
        if mode == "raw":
            if (alpha, num_clients, "raw", 0, ch, 1, 1) in completed:
                print(f"  skipping a={alpha} nc={num_clients} mode=raw k=0 (already done)")
            else:
                results.append(
                    run_single_experiment(
                        args, train_loaders, test_loaders, device, mode="raw", k=0
                    )
                )
        else:
            # Determine kernel sizes to sweep for fixed_random_conv
            ks_list = [ks_default]
            if mode == "fixed_random_conv" and getattr(args, "conv_kernel_size_list", None):
                ks_list = args.conv_kernel_size_list

            for cr in args.compression_ratio_list:
                k = activation_dim // cr
                for ks in ks_list:
                    if mode == "fixed_random_conv":
                        args.conv_kernel_size = ks
                    if (alpha, num_clients, mode, k, ch, ks if mode == "fixed_random_conv" else 1, getattr(args, "block_size", 1) if mode == "block_random" else 1) in completed:
                        print(f"  skipping a={alpha} nc={num_clients} mode={mode} k={k} ks={ks} (already done)")
                    else:
                        results.append(
                            run_single_experiment(
                                args, train_loaders, test_loaders, device, mode=mode, k=k
                            )
                        )
            # Restore default kernel size
            args.conv_kernel_size = ks_default

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
        activation_dim = cut_dims(args.dataset, getattr(args, "cut_channels", None))[3]
        k = 0 if args.mode == "raw" else activation_dim // args.compression_ratio
        ch = getattr(args, "client_heads", False)
        ks = getattr(args, "conv_kernel_size", 1)
        bs = getattr(args, "block_size", 1) if args.mode == "block_random" else 1
        if (args.alpha, args.num_clients, args.mode, k, ch, ks, bs) in completed:
            print(f"  skipping a={args.alpha} nc={args.num_clients} mode={args.mode} k={k} ks={ks} (already done)")
        else:
            run_single_experiment(
                args, train_loaders, test_loaders, device, mode=args.mode, k=k
            )


if __name__ == "__main__":
    main()