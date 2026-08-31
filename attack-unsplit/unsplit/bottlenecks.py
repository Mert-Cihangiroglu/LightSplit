"""Split-learning bottleneck modules ported from `utility/train_split.py`.

Three defenses, each as a self-contained `nn.Module` with `forward(h) -> (payload, info)`
and `lift_back(payload, info) -> recon`:

- `Learned1x1Bottleneck`   — spatial 1x1 conv compress + 1x1 conv lift-back.
- `FixedRandomBottleneck`  — fixed orthogonal projection `z = h @ R`; `R` is a buffer.
- `TopKSparseBottleneck`   — magnitude top-k_s sparsification; payload stays dense, lift is identity.
- `BlockRandomBottleneck` — block-local fixed orthonormal projection; spatial, privacy/utility dial.
- `RawBottleneck`          — identity (no compression); payload = flattened activation. For raw+DP.

`Learned1x1Bottleneck` operates on spatial tensors `(B, C, H, W)`; the other two on
flattened features `(B, dim)`. Callers (e.g. `DefendedSplitModel`) handle reshape.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


# Honest transmitted-byte accounting for sparse payloads (matched against a dense
# width-k projection): each kept coordinate costs a 4-byte float32 value + a 2-byte
# uint16 index. Ported from utility/train_split.py.
VALUE_BYTES = 4
INDEX_BYTES = 2


def _safe_svd(mat: torch.Tensor):
    """SVD with CPU fallback and NaN guard (ported as-is from train_split.py:40-48)."""
    if not torch.isfinite(mat).all():
        mat = torch.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        return torch.linalg.svd(mat, full_matrices=False)
    except torch._C._LinAlgError:
        u, s, vh = torch.linalg.svd(mat.cpu(), full_matrices=False)
        return u.to(mat.device), s.to(mat.device), vh.to(mat.device)


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


class TopKSparseBottleneck(nn.Module):
    """Magnitude-based top-k_s sparsification (matched-byte compression baseline).

    Keeps the `k_s` largest-magnitude coordinates per sample and zeroes the rest. The
    payload stays dense (`(B, dim)`) so the existing flatten/lift_back/reshape path is
    reused unchanged; `lift_back` is the identity because the payload already lives in
    activation space (no projection to undo). Gradients flow straight through the kept
    coordinates (mask-multiply). Emits no projected `z`, so WCC does not apply to it
    (consistent with utility/train_split.py).
    """

    def __init__(self, k_s: int):
        super().__init__()
        self.k_s = k_s

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        k = min(self.k_s, h.shape[1])
        _, idx = torch.topk(h.abs(), k, dim=1)
        mask = torch.zeros_like(h).scatter(1, idx, 1.0)   # 0/1 keep-mask, per sample
        return h * mask, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload


class BlockRandomBottleneck(nn.Module):
    """Block-local fixed orthonormal random projection (privacy/utility dial).

    Tiles the (C,H,W) cut tensor into block x block spatial tiles, flattens each
    tile to C*block*block dims, and projects it with a fixed orthonormal random R
    to out_per_block dims.  The macro grid (H/block x W/block) is preserved, so the
    random mixing is confined to a block x block receptive field -- the knob between
    privacy and utility:
        block = 1   -> per-position channel projection (max utility, min privacy)
        block = H   -> single global dense R == FixedRandomBottleneck (max privacy)
    Operates on the spatial tensor (like Learned1x1).  One R is shared across tiles
    and kept as a client-only buffer (never trained, never sent).  Transmitted payload
    = n_blocks * out_per_block floats (matched-byte ~= k).  Ported verbatim from
    utility/train_split.py.
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


class RawBottleneck(nn.Module):
    """No-compression identity bottleneck: the payload IS the flattened activation.

    Exists so the raw (no-defense) baseline can route through DefendedSplitModel and pick up
    the DP clip+noise applied in `_through_bottleneck` (the legacy `none` path has no bottleneck).
    Operates on flattened `(B, dim)` features; lift_back is the identity (caller reshapes).
    """

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return h, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 16, 16
    flat_dim = C * H * W
    K = 2048

    x_sp = torch.randn(B, C, H, W)
    bn = Learned1x1Bottleneck(C, 8)
    payload, info = bn(x_sp)
    recon = bn.lift_back(payload, info)
    assert payload.shape == (B, 8, H, W), payload.shape
    assert recon.shape == (B, C, H, W), recon.shape
    print(f"Learned1x1Bottleneck OK  payload={tuple(payload.shape)} recon={tuple(recon.shape)}")

    x_flat = x_sp.flatten(1)
    bn = FixedRandomBottleneck(flat_dim, K)
    payload, info = bn(x_flat)
    recon = bn.lift_back(payload, info)
    assert payload.shape == (B, K), payload.shape
    assert recon.shape == (B, flat_dim), recon.shape
    assert "bottleneck_z" in info, info.keys()
    print(f"FixedRandomBottleneck OK payload={tuple(payload.shape)} recon={tuple(recon.shape)}")

    K_S = max(1, (VALUE_BYTES * K) // (VALUE_BYTES + INDEX_BYTES))
    bn = TopKSparseBottleneck(K_S)
    payload, info = bn(x_flat)
    recon = bn.lift_back(payload, info)
    assert payload.shape == (B, flat_dim), payload.shape
    assert recon.shape == (B, flat_dim), recon.shape
    assert info == {}, info
    nnz = (payload != 0).sum(dim=1)
    assert bool((nnz == K_S).all()), nnz.tolist()
    print(f"TopKSparseBottleneck OK  payload={tuple(payload.shape)} recon={tuple(recon.shape)} k_s={K_S} nnz/row={K_S}")

    BLOCK = 2
    bn = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W, block=BLOCK, k=K)
    payload, info = bn(x_sp)
    recon = bn.lift_back(payload, info)
    n_blocks = (H // BLOCK) * (W // BLOCK)
    out_per_block = max(1, K // n_blocks)
    assert payload.shape == (B, n_blocks, out_per_block), payload.shape
    assert recon.shape == (B, C, H, W), recon.shape
    assert "bottleneck_z" in info and info["bottleneck_z"].shape == (B, n_blocks * out_per_block), info
    print(f"BlockRandomBottleneck OK payload={tuple(payload.shape)} recon={tuple(recon.shape)} "
          f"n_blocks={n_blocks} out_per_block={out_per_block}")
