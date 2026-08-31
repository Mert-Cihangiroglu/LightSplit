"""
LightSplit core, ported verbatim from the image pipeline to a tabular MLP.

The method lives entirely in *activation space*, so nothing here is image- or
modality-specific:

  * fixed random projection  R : R^d -> R^k   (Johnson-Lindenstrauss bottleneck)
  * server lift-back         phi: R^k -> R^d   (fixed  R z~ , or learned MLP)
  * Within-Class Compaction loss on the transmitted z~

These mirror FixedRandomBottleneck / FixedRandomMLPLiftBottleneck and the WCC
term in utility/train_split.py. Only the encoder (head/backbone/tail)
changes: an MLP over the 561 HAR features instead of a CNN over pixels.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bottlenecks ──────────────────────────────────────────────────────────────
class FixedRandomBottleneck(nn.Module):
    """LightSplit-F: fixed orthonormal projection + fixed linear lift-back.

    R has orthonormal columns (R^T R = I_k), so lift-back R R^T z is the
    orthogonal projection of z onto a random k-dim subspace. Parameter-free.
    """

    def __init__(self, activation_dim: int, k: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(activation_dim, k))
        self.register_buffer("R", q)          # (d, k), client-side only
        self.k = k

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        ztilde = z @ self.R                    # (B, k) -- the transmitted payload
        return ztilde, {"bottleneck_z": ztilde}

    def lift_back(self, payload: torch.Tensor) -> torch.Tensor:
        return payload @ self.R.T              # (B, d)


class LearnedLiftBottleneck(nn.Module):
    """LightSplit-L: fixed projection on the client + learned MLP lift-back.

    The client still compresses with the fixed R; the server learns a nonlinear
    map from the k-dim payload back to d dims. The server never sees R.
    """

    def __init__(self, activation_dim: int, k: int, hidden: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(activation_dim, k))
        self.register_buffer("R", q)
        self.k = k
        self.learned_lift = nn.Sequential(
            nn.Linear(k, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, activation_dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        ztilde = z @ self.R
        return ztilde, {"bottleneck_z": ztilde}

    def lift_back(self, payload: torch.Tensor) -> torch.Tensor:
        return self.learned_lift(payload)


class RawBottleneck(nn.Module):
    """No-defense control: transmit the full cut-layer activation (no projection).

    Used as the baseline that isolates what LightSplit actually buys. WCC can
    still be applied on the raw activation, but with k == d there is no
    compression and no JL bottleneck.
    """

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, {"bottleneck_z": z}

    def lift_back(self, payload: torch.Tensor) -> torch.Tensor:
        return payload


# ── Baselines for the comparison sweep ───────────────────────────────────────
class LearnedCodecBottleneck(nn.Module):
    """Learned-codec baseline: a *learned* linear encoder d->k on the client and a
    learned MLP decoder k->d on the server (cf. learned_1x1 in the image pipeline).

    Contrast with LightSplit: same payload budget k, but the client spends trainable
    parameters to learn the compressor, whereas LightSplit's projection is fixed and
    parameter-free. Matched decoder capacity to LightSplit-L for a fair comparison.
    """

    def __init__(self, activation_dim: int, k: int, hidden: int):
        super().__init__()
        self.encoder = nn.Linear(activation_dim, k)            # client-side, learned
        self.decoder = nn.Sequential(                          # server-side, learned
            nn.Linear(k, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, activation_dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        ztilde = self.encoder(z)
        return ztilde, {"bottleneck_z": ztilde}

    def lift_back(self, payload: torch.Tensor) -> torch.Tensor:
        return self.decoder(payload)


# Honest transmitted-byte accounting for sparse payloads: each kept coordinate
# costs a 4-byte float32 value + a 2-byte uint16 index (copied from train_split.py).
VALUE_BYTES = 4
INDEX_BYTES = 2


class TopKSparseBottleneck(nn.Module):
    """Magnitude-based top-k sparsification (matched-byte compression baseline).

    Keeps the k_s largest-magnitude coordinates per sample and zeroes the rest;
    payload stays dense (d-dim) so lift-back is identity. k_s is chosen so the
    honest byte cost k_s*(4+2) matches LightSplit's k float32 scalars. Gradients
    flow straight through the kept coordinates (mask-multiply). Not a LightSplit
    component, so WCC is not applied. Verbatim port of train_split.py:139-162.
    """

    def __init__(self, k_s: int):
        super().__init__()
        self.k_s = k_s

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        bsz, dim = h.shape
        k = min(self.k_s, dim)
        _, idx = torch.topk(h.abs(), k, dim=1)
        mask = torch.zeros_like(h).scatter(1, idx, 1.0)
        return h * mask, {}

    def lift_back(self, payload: torch.Tensor) -> torch.Tensor:
        return payload


# ── Within-Class Compaction ──────────────────────────────────────────────────
def wcc_loss(ztilde: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum over classes of mean within-class squared distance to the centroid.

    Suppresses I(x; z~ | y): the per-sample signal that survives once the label
    is fixed, which is exactly what reconstruction / re-identification attacks
    exploit. Computed on the client; labels never leave. (train_split.py:724-734)
    """
    loss = ztilde.new_tensor(0.0)
    for c in torch.unique(y):
        mask = y == c
        if mask.sum() > 1:
            zc = ztilde[mask]
            mu = zc.mean(dim=0, keepdim=True)
            loss = loss + (zc - mu).pow(2).sum(dim=1).mean()
    return loss


# ── U-shaped split model ─────────────────────────────────────────────────────
class HARSplitModel(nn.Module):
    """U-shaped SL model for tabular HAR.

    client head f  : 561 -> d   (2-layer MLP; nonlinear so the cut layer is not
                                 trivially invertible by a pseudo-inverse)
    [bottleneck]   : d  -> k    transmit  -> k -> d  lift-back
    server backbone: d -> 128
    client tail h  : 128 -> C
    """

    def __init__(
        self,
        in_features: int = 561,
        d: int = 256,
        k: int = 32,
        num_classes: int = 6,
        liftback: str = "learned",
        hidden: int = 256,
        dp_clip: float = 0.0,
        dp_sigma: float = 0.0,
    ):
        super().__init__()
        # DP on the TRANSMITTED payload: per-sample L2 clip (bounds sensitivity) then
        # Gaussian noise with std = dp_sigma * dp_clip. Same recipe as the FORA / UnSplit
        # / SDAR DP arms. Applied on every forward, training AND attack/eval, so the
        # server and the attacker both receive a noised payload. Gated on dp_clip > 0, so
        # the default path is byte-identical to the un-noised model.
        self.dp_clip, self.dp_sigma = float(dp_clip), float(dp_sigma)
        self.d, self.mode = d, liftback
        self.liftback = liftback  # back-compat alias
        # "none" is the no-defense control: payload is the full d-dim activation.
        self.k = d if liftback == "none" else k
        self._k_s = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))

        self.head = nn.Sequential(
            nn.Linear(in_features, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
        )

        if liftback == "none":
            self.bottleneck: nn.Module = RawBottleneck()                 # vanilla SL
        elif liftback == "fixed":
            self.bottleneck = FixedRandomBottleneck(d, k)                # LightSplit-F
        elif liftback == "learned":
            self.bottleneck = LearnedLiftBottleneck(d, k, hidden)        # LightSplit-L
        elif liftback == "codec":
            self.bottleneck = LearnedCodecBottleneck(d, k, hidden)       # learned codec
        elif liftback == "topk_sparse":
            self.bottleneck = TopKSparseBottleneck(self._k_s)            # top-k baseline
        else:
            raise ValueError(f"unknown liftback: {liftback}")

        self.backbone = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.tail = nn.Linear(128, num_classes)

    def _dp(self, payload: torch.Tensor) -> torch.Tensor:
        """Per-sample L2 clip to dp_clip, then Gaussian noise std = dp_sigma * dp_clip."""
        if self.dp_clip <= 0.0:
            return payload
        norms = payload.norm(dim=1, keepdim=True).clamp(min=1e-12)
        payload = payload * torch.clamp(self.dp_clip / norms, max=1.0)
        if self.dp_sigma > 0.0:
            payload = payload + torch.randn_like(payload) * (self.dp_sigma * self.dp_clip)
        return payload

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.head(x)
        ztilde, _ = self.bottleneck(z)             # transmitted payload (B, k)
        ztilde = self._dp(ztilde)                  # DP applied to what is transmitted
        zhat = self.bottleneck.lift_back(ztilde)   # server reconstructs (B, d)
        u = self.backbone(zhat)
        return self.tail(u), ztilde

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Server-visible payload z~ = R^T f(x). This is the attacker's input."""
        self.eval()
        return self._dp(self.bottleneck(self.head(x))[0])

    @property
    def comm_floats_per_sample(self) -> float:
        """Honest forward payload size, in float32-equivalents per sample."""
        if self.mode == "none":
            return float(self.d)
        if self.mode == "topk_sparse":
            return float((self._k_s * (VALUE_BYTES + INDEX_BYTES)
                          + VALUE_BYTES - 1) // VALUE_BYTES)
        return float(self.k)  # fixed / learned / codec all send k scalars

    @property
    def compression_ratio(self) -> float:
        return self.d / self.comm_floats_per_sample


# ── Canonical tabular MIA classifier (Nasr, Shokri & Houmansadr, S&P'19) ───────
# The standard Purchase-100 / Texas-100 target model in the membership-inference
# literature: a 4-hidden-layer fully-connected net [1024, 512, 256, 128] with Tanh
# (no BatchNorm), softmax over 100 classes. LightSplit's cut is the 256-wide layer
# (d=256): client head = in->1024->512->256, server backbone = 256->128, tail->C.
ARCH = {
    "purchase100": {"head_dims": [1024, 512], "d": 256, "backbone_dims": [128]},
    "texas100":    {"head_dims": [1024, 512], "d": 256, "backbone_dims": [128]},
}


class TabularSplitModel(nn.Module):
    """U-shaped split model with a configurable MLP trunk (Tanh, no BatchNorm) so
    each dataset can use its recognized classifier. The LightSplit bottleneck sits
    at the cut dim d (transmit k<<d), identical in role to HARSplitModel."""

    def __init__(self, in_features: int, num_classes: int, head_dims, d: int,
                 backbone_dims, liftback: str = "none", k: int = 32, hidden: int = 256):
        super().__init__()
        self.d, self.mode = d, liftback
        self.liftback = liftback
        self.k = d if liftback == "none" else k
        self._k_s = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))

        layers, prev = [], in_features
        for h in [*head_dims, d]:                       # client head: in -> ... -> d
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.head = nn.Sequential(*layers)

        if liftback == "none":
            self.bottleneck: nn.Module = RawBottleneck()
        elif liftback == "fixed":
            self.bottleneck = FixedRandomBottleneck(d, k)
        elif liftback == "learned":
            self.bottleneck = LearnedLiftBottleneck(d, k, hidden)
        elif liftback == "codec":
            self.bottleneck = LearnedCodecBottleneck(d, k, hidden)
        elif liftback == "topk_sparse":
            self.bottleneck = TopKSparseBottleneck(self._k_s)
        else:
            raise ValueError(f"unknown liftback: {liftback}")

        layers, prev = [], d                            # server backbone: d -> ...
        for h in backbone_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.tail = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.head(x)
        ztilde, _ = self.bottleneck(z)
        zhat = self.bottleneck.lift_back(ztilde)
        u = self.backbone(zhat)
        return self.tail(u), ztilde

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.bottleneck(self.head(x))[0]

    @property
    def comm_floats_per_sample(self) -> float:
        if self.mode == "none":
            return float(self.d)
        if self.mode == "topk_sparse":
            return float((self._k_s * (VALUE_BYTES + INDEX_BYTES)
                          + VALUE_BYTES - 1) // VALUE_BYTES)
        return float(self.k)

    @property
    def compression_ratio(self) -> float:
        return self.d / self.comm_floats_per_sample
