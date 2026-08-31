"""Per-tensor activation stats for backdoor detection.

Ports moment / norm / spectral / cosine helpers from
an internal reference implementation. Trimmed to the metric set
specified in the plan: moments (mean/var/skew/kurt), norms (l1/l2/linf),
spectral (top_sv / spectral_energy_ratio / effective_rank), and
cosine_to_global. No sparsity, no entropy.

Every function expects a (B, D) tensor; spatial activations should be flattened
with .flatten(1) before being passed in.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def compute_moment_stats(t: torch.Tensor, prefix: str = "z") -> Dict[str, float]:
    """mean / var / skewness / excess kurtosis over all elements of t."""
    vals = t.detach().float()
    mean = vals.mean().item()
    var = vals.var().item()

    centered = vals - vals.mean()
    std = vals.std().clamp(min=1e-10)
    normed = centered / std
    skewness = normed.pow(3).mean().item()
    kurtosis = normed.pow(4).mean().item() - 3.0  # excess kurtosis

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_var": var,
        f"{prefix}_skewness": skewness,
        f"{prefix}_kurtosis": kurtosis,
    }


def compute_norm_stats(t: torch.Tensor, prefix: str = "z") -> Dict[str, float]:
    """L1 / L2 / Linf norms per sample, batch-averaged. Expects (B, D)."""
    vals = t.detach().float()
    l1 = vals.abs().sum(dim=1).mean().item()
    l2 = vals.norm(dim=1).mean().item()
    linf = vals.abs().max(dim=1).values.mean().item()

    return {
        f"{prefix}_l1_norm": l1,
        f"{prefix}_l2_norm": l2,
        f"{prefix}_linf_norm": linf,
    }


def _safe_svd(mat: torch.Tensor):
    if not torch.isfinite(mat).all():
        mat = torch.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        return torch.linalg.svd(mat.float(), full_matrices=False)
    except torch._C._LinAlgError:
        u, s, vh = torch.linalg.svd(mat.float().cpu(), full_matrices=False)
        return u.to(mat.device), s.to(mat.device), vh.to(mat.device)


def compute_spectral_stats(t: torch.Tensor, prefix: str = "z") -> Dict[str, float]:
    """SVD-based spectral metrics on a (B, D) batch matrix.

    Returns top_sv, spectral_energy_ratio, effective_rank.
    """
    vals = t.detach().float()
    if vals.shape[0] < 2 or vals.shape[1] < 2:
        return {
            f"{prefix}_top_sv": 0.0,
            f"{prefix}_spectral_energy_ratio": 0.0,
            f"{prefix}_effective_rank": 1.0,
        }

    _, s, _ = _safe_svd(vals)
    s = s.clamp(min=0)

    top_sv = s[0].item()
    s_sq = s ** 2
    total_energy = s_sq.sum().item()
    energy_ratio = (s_sq[0].item() / total_energy) if total_energy > 1e-10 else 0.0

    if total_energy > 1e-10:
        p = s_sq / total_energy
        p = p.clamp(min=1e-10)
        entropy = -(p * p.log()).sum().item()
        effective_rank = math.exp(entropy)
    else:
        effective_rank = 1.0

    return {
        f"{prefix}_top_sv": top_sv,
        f"{prefix}_spectral_energy_ratio": energy_ratio,
        f"{prefix}_effective_rank": effective_rank,
    }


def compute_cosine_to_global(
    client_mean: torch.Tensor,
    global_mean: torch.Tensor,
    prefix: str = "z",
) -> Dict[str, float]:
    """Cosine of two 1D vectors. Returns 0.0 if either vector is ~zero."""
    if client_mean.norm() < 1e-10 or global_mean.norm() < 1e-10:
        return {f"{prefix}_cosine_to_global": 0.0}

    cos = F.cosine_similarity(
        client_mean.unsqueeze(0).float(),
        global_mean.unsqueeze(0).float(),
    ).item()
    return {f"{prefix}_cosine_to_global": cos}
