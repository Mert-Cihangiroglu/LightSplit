"""LightSplit bottlenecks, text-shaped.

Ported verbatim (mechanism-wise) from the production image harness
utility/train_split.py:
  - FixedRandomBottleneck            (line 166)  -> LightSplit-F
  - FixedRandomMLPLiftBottleneck     (line 202)  -> LightSplit-L

The only change is that these operate on transformer hidden states h : [B, T, d]
instead of a flat [B, d] activation. The fixed projection R : [d, k] broadcasts
over the leading (B, T) dims for free (h @ R -> [B, T, k]); only the learned MLP
lift (L) reshapes to [B*T, k] because BatchNorm1d/Linear want a 2-D batch.

No WCC and no DP here (this build is a utility-only generality demo).
R is a public, server-known, no-grad buffer.
"""
import torch
import torch.nn as nn


class RawBottleneck(nn.Module):
    """Identity bottleneck (utility ceiling): no compression."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.k = d
        self.compression_ratio = 1.0

    def project(self, h):
        return h

    def lift(self, z):
        return z

    def forward(self, h):
        return h


class FixedRandomBottleneck(nn.Module):
    """LightSplit-F: fixed public orthonormal projection R (d->k), transpose lift-back."""

    def __init__(self, d: int, k: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(d, k))  # reduced -> q : [d, k], columns orthonormal
        self.register_buffer("R", q[:, :k])
        self.d = d
        self.k = k
        self.compression_ratio = d / k

    def project(self, h):
        return h @ self.R          # [..., k]

    def lift(self, z):
        return z @ self.R.T        # [..., d]  (rank-k reconstruction)

    def forward(self, h):
        return self.lift(self.project(h))


class FixedRandomMLPLiftBottleneck(nn.Module):
    """LightSplit-L: fixed R projection (client) + learned MLP lift-back (server).

    R is client-side/public; the server MLP has no access to R and learns to map
    the k-dim payload back to the d-dim activation. Mirrors the production
    Sequential(Linear(k, hidden), BatchNorm1d(hidden), ReLU, Linear(hidden, d)).
    """

    def __init__(self, d: int, k: int, hidden: int = 512, norm: str = "batch"):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(d, k))
        self.register_buffer("R", q[:, :k])
        self.d = d
        self.k = k
        self.compression_ratio = d / k
        norm_layer = nn.BatchNorm1d(hidden) if norm == "batch" else nn.LayerNorm(hidden)
        self.learned_lift = nn.Sequential(
            nn.Linear(k, hidden),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d),
        )

    def project(self, h):
        return h @ self.R          # [..., k]

    def lift(self, z):
        shape = z.shape                       # [..., k]
        out = self.learned_lift(z.reshape(-1, self.k))
        return out.reshape(*shape[:-1], self.d)

    def forward(self, h):
        return self.lift(self.project(h))


def build_bottleneck(mode: str, d: int, cr: float, mlp_hidden: int = 512, norm: str = "batch"):
    if mode == "raw":
        return RawBottleneck(d)
    k = max(1, round(d / cr))
    if mode == "fixed_random":
        return FixedRandomBottleneck(d, k)
    if mode == "fixed_random_ll":
        return FixedRandomMLPLiftBottleneck(d, k, hidden=mlp_hidden, norm=norm)
    raise ValueError(f"unknown bottleneck mode: {mode}")
