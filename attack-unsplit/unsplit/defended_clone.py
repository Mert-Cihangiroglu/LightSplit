"""DefendedClone: architecture-faithful clone for UnSplit attack against a defended victim.

The clone is a fresh `MnistNet`/`CifarNet` head (random init) plus a learnable `compress`
matching the *type* of the victim's bottleneck. `forward(x, end=split_layer)` returns the
post-compress payload — same shape as `DefendedSplitModel.compress_payload(x)` — so that
`unsplit.attacks.model_inversion_stealing` can be reused unchanged.

Threat model: attacker knows the head architecture and the bottleneck *type* and `k`, but not
the trained weights, R, or any captured SVD basis. UnSplit alternates between optimizing
`x_pred` and the clone's parameters (head + compress).

the SVD basis degenerates (`effective_k = max(1, bsz - 1) = 1`), and the standard UnSplit
"compression-family known, weights unknown" assumption fits a learnable linear naturally.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DefendedClone(nn.Module):
    def __init__(self, model_class, split_layer: int, defense: str, k: int,
                 in_channels: int, spatial_hw: Tuple[int, int], block_size: int = 2):
        super().__init__()
        self.split_layer = split_layer
        self.defense = defense
        self.head = model_class()
        H, W = spatial_hw
        activation_dim = in_channels * H * W
        if defense == "learned_1x1":
            k_channels = max(in_channels // (activation_dim // k), 1)
            self.compress: nn.Module = nn.Conv2d(in_channels, k_channels, 1)
        elif defense == "fixed_random":
            self.compress = nn.Linear(activation_dim, k, bias=False)
        elif defense == "block_random":
            # The victim projects each block_size x block_size tile by one shared fixed R.
            # A stride-block conv is exactly that op with weights shared across tiles, so
            # it's the natural learnable clone (weights unknown, family known). Payload is
            # 3D (B, n_blocks, out_per_block); forward reshapes the conv output to match.
            n_blocks = (H // block_size) * (W // block_size)
            out_per_block = max(1, k // n_blocks)
            self.compress = nn.Conv2d(in_channels, out_per_block,
                                      kernel_size=block_size, stride=block_size)
        elif defense in ("topk_sparse", "raw", "none"):
            # topk_sparse / raw payload IS the (DP-noised) activation, dim = activation_dim, no
            # projection to clone: the attacker matches the observed coords, so the clone just
            # emits the dense flattened head output. (raw+DP: the victim clips+noises this.)
            self.compress = nn.Identity()
        else:
            raise ValueError(f"unknown defense: {defense}")

    def forward(self, x: torch.Tensor, start: int = 0, end: int = 10) -> torch.Tensor:
        h = self.head(x, start=start, end=end)
        if end != self.split_layer or self.defense == "none":
            return h
        if self.defense == "learned_1x1":
            return self.compress(h)
        if self.defense == "block_random":
            # conv -> (B, out_per_block, grid_h, grid_w); reshape to the victim's
            # (B, n_blocks, out_per_block), row-major over (grid_h, grid_w).
            return self.compress(h).flatten(2).transpose(1, 2)
        return self.compress(h.flatten(1))


def build_target_mask(target: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a binary mask zeroing the structurally-zero columns of a payload, so the
    inversion MSE only scores observed coordinates. Two cases use it:
      - `topk_sparse`: the payload keeps only the top-k_s magnitude coords, zeroing the rest.
    Returns None when no zero columns are detected — caller can use `None` as a no-op."""
    if target.dim() != 2:
        return None
    col_active = (target.abs().sum(dim=0) > 0).float()
    if col_active.sum().item() == col_active.numel():
        return None
    return col_active.unsqueeze(0)


if __name__ == "__main__":
    from .defended_models import DefendedSplitModel
    from .models import CifarNet

    torch.manual_seed(0)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    for defense in ("learned_1x1", "fixed_random", "topk_sparse", "block_random"):
        victim = DefendedSplitModel(CifarNet, split_layer=4, defense=defense, k=2048,
                                    in_channels=64, spatial_hw=(16, 16))
        clone = DefendedClone(CifarNet, split_layer=4, defense=defense, k=2048,
                              in_channels=64, spatial_hw=(16, 16))
        v_payload = victim.compress_payload(x)
        c_payload = clone(x, end=4)
        assert v_payload.shape == c_payload.shape, f"{defense}: v={v_payload.shape} c={c_payload.shape}"
        print(f"{defense:14s} MATCH \u2713  victim={tuple(v_payload.shape)} clone={tuple(c_payload.shape)}")
