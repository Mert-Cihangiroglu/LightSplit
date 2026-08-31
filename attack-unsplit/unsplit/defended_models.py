"""DefendedSplitModel: client/server split + bottleneck defense.

Wraps a `MnistNet` or `CifarNet` pair (separate client and server instances) plus one of the
bottleneck modules from `bottlenecks.py`. Mirrors the no-defense flow of
`model_inversion_demo.py` but routes the smashed activation through a compress + lift_back step
between the client's `end=split_layer` cut and the server's `start=split_layer+1` continuation.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .bottlenecks import (
    BlockRandomBottleneck,
    FixedRandomBottleneck,
    Learned1x1Bottleneck,
    RawBottleneck,
    TopKSparseBottleneck,
    VALUE_BYTES,
    INDEX_BYTES,
)


def build_bottleneck(defense: str, k: int, in_channels: int, spatial_hw: Tuple[int, int],
                     block_size: int = 2) -> nn.Module:
    """Construct a bottleneck for one of the supported defenses.

    `k` is the payload-size budget in scalars/sample. For learned_1x1 the payload is spatial
    `(B, k_channels, H, W)`, so `k_channels` is back-solved from `k` and the spatial shape
    so the total scalar count matches. For block_random, `block_size` sets the tile size
    (the privacy/utility dial) and `out_per_block` is back-solved from `k` and the grid.
    """
    H, W = spatial_hw
    activation_dim = in_channels * H * W
    if defense == "learned_1x1":
        k_channels = max(in_channels // (activation_dim // k), 1)
        return Learned1x1Bottleneck(in_channels=in_channels, k_channels=k_channels)
    if defense == "fixed_random":
        return FixedRandomBottleneck(activation_dim=activation_dim, k=k)
    if defense == "topk_sparse":
        # matched-byte: keep k_s coords so k_s*(4+2) bytes ~= k*4 bytes (the dense-k budget)
        k_s = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))
        return TopKSparseBottleneck(k_s=k_s)
    if defense == "block_random":
        return BlockRandomBottleneck(channels=in_channels, hw_h=H, hw_w=W,
                                     block=block_size, k=k)
    if defense == "raw":
        return RawBottleneck()  # no compression; payload = flattened activation (for raw+DP)
    raise ValueError(f"unknown defense: {defense}")


class DefendedSplitModel(nn.Module):
    def __init__(self, model_class, split_layer: int, defense: str, k: int,
                 in_channels: int, spatial_hw: Tuple[int, int], block_size: int = 2,
                 clip_norm: float = 0.0, noise_multiplier: float = 0.0):
        super().__init__()
        self.client = model_class()
        self.server = model_class()
        self.split_layer = split_layer
        self.defense = defense
        self.bottleneck = build_bottleneck(defense, k, in_channels, spatial_hw, block_size)
        # DP on the transmitted payload: per-sample L2 clip (bounds sensitivity) then calibrated
        # Gaussian noise (std = noise_multiplier * clip_norm). Applied train AND eval so the server
        # and the attacker truly receive a noised representation. Ported from
        # utility/train_split.py:_apply_dp. clip_norm<=0 => DP off.
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def _apply_dp(self, payload: torch.Tensor) -> torch.Tensor:
        """Per-sample L2 clip + Gaussian noise on the transmitted payload (any rank; flatten(1))."""
        if self.clip_norm <= 0:
            return payload
        flat = payload.flatten(1)
        norms = flat.norm(dim=1, keepdim=True)
        clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
        flat = flat * clip_factor
        if self.noise_multiplier > 0:
            flat = flat + torch.randn_like(flat) * (self.noise_multiplier * self.clip_norm)
        return flat.reshape_as(payload)

    def _through_bottleneck(self, smashed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if isinstance(self.bottleneck, (Learned1x1Bottleneck, BlockRandomBottleneck)):
            payload, info = self.bottleneck(smashed)
            payload = self._apply_dp(payload)
            recon = self.bottleneck.lift_back(payload, info)
            return payload, recon, info
        B, C, H, W = smashed.shape
        flat = smashed.flatten(1)
        payload, info = self.bottleneck(flat)
        payload = self._apply_dp(payload)
        recon = self.bottleneck.lift_back(payload, info).view(B, C, H, W)
        return payload, recon, info

    def full_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        smashed = self.client(x, end=self.split_layer)
        _, recon, info = self._through_bottleneck(smashed)
        logits = self.server(recon, start=self.split_layer + 1)
        return logits, info

    def compress_payload(self, x: torch.Tensor) -> torch.Tensor:
        smashed = self.client(x, end=self.split_layer)
        payload, _, _ = self._through_bottleneck(smashed)
        return payload


if __name__ == "__main__":
    from .models import CifarNet

    torch.manual_seed(0)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    expected_payload = {
        "learned_1x1":  (B, 8, 16, 16),
        "fixed_random": (B, 2048),
        "topk_sparse":  (B, 64 * 16 * 16),  # dense payload; only k_s coords nonzero
        "block_random": (B, 64, 32),        # block=2: grid 8x8 -> n_blocks=64, out_per_block=32
    }
    # Defenses that emit a flattened projected z for WCC.
    emits_z = {"fixed_random", "block_random"}
    for defense, exp_shape in expected_payload.items():
        model = DefendedSplitModel(CifarNet, split_layer=4, defense=defense, k=2048,
                                   in_channels=64, spatial_hw=(16, 16))
        logits, info = model.full_forward(x)
        payload = model.compress_payload(x)
        assert logits.shape == (B, 10), f"{defense}: logits {logits.shape}"
        assert payload.shape == exp_shape, f"{defense}: payload {payload.shape}"
        if defense in emits_z:
            assert "bottleneck_z" in info and info["bottleneck_z"].shape == (B, 2048), info
        else:
            assert "bottleneck_z" not in info, info
        print(f"{defense:14s} MATCH \u2713  logits={tuple(logits.shape)} payload={tuple(payload.shape)}")
