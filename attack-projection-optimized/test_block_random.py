#!/usr/bin/env python
"""Standalone CPU test for the privileged block_random port.

Run:  python test_block_random.py
Exits 0 on success, nonzero on any failure (suitable as a commit gate).

The decisive check (test 1) proves the flat full-R formulation used here is
NUMERICALLY IDENTICAL to the unfold-based BlockRandomBottleneck shipped in
utility/train_split.py (and ported to TF in SDAR): same z, same lift.
The remaining checks prove the full R is orthonormal, has the right block
locality (the privacy/utility dial), reduces to fixed_random at block=H, and
that the whole known-R attack surface (model.bottleneck.R, pinv, oracle,
known-R decoder, grad fusion) works unchanged on the block-structured R.
"""
import sys

import torch

from train_split import BlockRandomBottleneck, SplitModel

SEED = 1234
torch.use_deterministic_algorithms(False)


# --- faithful copy of the utility harness's unfold tiling (the reference algorithm) ----
def to_blocks(z, block, grid_h, grid_w, in_per_block):
    b = z.shape[0]
    x = z.unfold(2, block, block).unfold(3, block, block)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()          # (B, grid_h, grid_w, C, blk, blk)
    return x.reshape(b, grid_h * grid_w, in_per_block)


def from_blocks(x, block, grid_h, grid_w, channels, H, W):
    b = x.shape[0]
    x = x.reshape(b, grid_h, grid_w, channels, block, block)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()          # (B, C, grid_h, blk, grid_w, blk)
    return x.reshape(b, channels, H, W)


def build_with_known_basis(C, H, W, block, k):
    """Construct the bottleneck and independently reproduce its shared basis R_b.

    __init__ draws RNG exactly once (torch.randn(in_per_block, out_per_block)) before
    the QR, so re-seeding identically reproduces R_b bit-for-bit."""
    torch.manual_seed(SEED)
    bn = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W, block=block, k=k)
    torch.manual_seed(SEED)
    rb = torch.linalg.qr(torch.randn(bn.in_per_block, bn.out_per_block))[0]
    return bn, rb


def test_exact_vs_unfold():
    """z and lift from the flat full-R path == the unfold reference, to fp tol."""
    cases = [
        (8, 4, 4, 1, 32), (8, 4, 4, 2, 32), (8, 4, 4, 4, 32),
        (64, 8, 8, 1, 512), (64, 8, 8, 2, 512), (64, 8, 8, 4, 512), (64, 8, 8, 8, 512),
        (64, 8, 8, 2, 256), (64, 8, 8, 4, 128),
    ]
    for C, H, W, block, k in cases:
        bn, rb = build_with_known_basis(C, H, W, block, k)
        torch.manual_seed(SEED + 1)
        h_spatial = torch.randn(3, C, H, W)
        h_flat = h_spatial.flatten(1)

        z, info = bn(h_flat)
        ref_payload = to_blocks(h_spatial, block, bn.grid_h, bn.grid_w, bn.in_per_block) @ rb
        ref_z = ref_payload.flatten(1)
        assert z.shape == ref_z.shape == (3, bn.payload_floats), (z.shape, ref_z.shape)
        assert torch.allclose(z, ref_z, atol=1e-5), \
            f"forward mismatch C{C} H{H} block{block} k{k}: {(z - ref_z).abs().max()}"

        lift = bn.lift_back(z, info)
        ref_lift = from_blocks(ref_payload @ rb.t(), block, bn.grid_h, bn.grid_w, C, H, W).flatten(1)
        assert torch.allclose(lift, ref_lift, atol=1e-5), \
            f"lift mismatch C{C} H{H} block{block} k{k}: {(lift - ref_lift).abs().max()}"
        # bottleneck_z stored for WCC / grad fusion is exactly the transmitted z
        assert torch.equal(info["bottleneck_z"], z)
    print("  [ok] test_exact_vs_unfold: flat full-R == unfold reference (9 cases)")


def test_orthonormal_columns():
    """R has orthonormal columns -> z@R.T is a true rank-payload projector."""
    for block in (1, 2, 4, 8):
        bn, _ = build_with_known_basis(64, 8, 8, block, 512)
        gram = bn.R.t() @ bn.R
        eye = torch.eye(bn.payload_floats)
        assert torch.allclose(gram, eye, atol=1e-5), \
            f"R not orthonormal at block={block}: {(gram - eye).abs().max()}"
        # projector P = R R^T is idempotent
        P = bn.R @ bn.R.t()
        assert torch.allclose(P @ P, P, atol=1e-5), f"P not idempotent at block={block}"
    print("  [ok] test_orthonormal_columns: R.T@R==I and P^2==P (block 1/2/4/8)")


def test_block_locality_dial():
    """block=1 mixes only within one spatial position; block=H mixes globally."""
    HW = 8 * 8
    bn1, _ = build_with_known_basis(64, 8, 8, 1, 512)
    # every output column touches exactly one (y,x): all its nonzero rows share y*W+x
    for m in (0, 7, 100, bn1.payload_floats - 1):
        rows = torch.nonzero(bn1.R[:, m]).flatten()
        assert rows.numel() == 64, f"block=1 col {m} should touch all 64 channels, got {rows.numel()}"
        yx = rows % HW                                    # flat index = c*HW + (y*W+x)
        assert (yx == yx[0]).all(), f"block=1 col {m} spans multiple positions {yx.unique().tolist()}"

    bn8, _ = build_with_known_basis(64, 8, 8, 8, 512)
    assert bn8.n_blocks == 1, "block=8 on an 8x8 cut must be a single global tile"
    assert bn8.R.shape == (4096, 512), bn8.R.shape
    # a global dense R: a representative column touches far more than one position
    col0_rows = torch.nonzero(bn8.R[:, 0]).flatten()
    assert (col0_rows % HW).unique().numel() > 1, "block=8 column should span many positions"
    print("  [ok] test_block_locality_dial: block=1 local (1 pos), block=8 global")


def test_payload_constant_across_blocks():
    """At a fixed CR the transmitted payload is held constant across block sizes."""
    for k in (512, 256, 128):
        widths = {build_with_known_basis(64, 8, 8, b, k)[0].payload_floats for b in (1, 2, 4, 8)}
        assert widths == {k}, f"payload not constant at k={k}: {widths}"
    print("  [ok] test_payload_constant_across_blocks: payload==k for block 1/2/4/8 at CR8/16/32")


def test_splitmodel_integration():
    """End-to-end SplitModel: forward runs, R is exposed full-size, grad path works."""
    for block in (1, 2, 8):
        torch.manual_seed(SEED)
        model = SplitModel(dataset="cifar10", mode="block_random", k=512,
                           deep_head=True, block_size=block).eval()
        assert model.bottleneck.R.shape == (4096, 512), model.bottleneck.R.shape
        assert model.payload_dim == 512, model.payload_dim

        x = torch.randn(4, 3, 32, 32)
        logits, info = model(x)
        assert logits.shape == (4, 10) and torch.isfinite(logits).all(), logits.shape
        assert info["bottleneck_z"].shape == (4, 512), info["bottleneck_z"].shape

        # returned-gradient path used by Exp 6 / WCC must differentiate bottleneck_z
        z = info["bottleneck_z"]
        loss = logits.sum()
        g = torch.autograd.grad(loss, z, retain_graph=False)[0]
        assert g.shape == (4, 512) and torch.isfinite(g).all() and g.abs().sum() > 0, \
            "returned gradient dL/dz is degenerate"
    print("  [ok] test_splitmodel_integration: forward + full R + dL/dz (block 1/2/8)")


def test_known_r_attack_surface():
    """The known-R attack decoders accept the block-structured R unchanged."""
    from privileged_attack import (KnownRImageDecoder, OracleActivationAttacker,
                                    GradFusionImageDecoder)
    bn, _ = build_with_known_basis(64, 8, 8, 2, 512)
    R, d, k = bn.R, 4096, bn.payload_floats
    z = torch.randn(5, k)
    g = torch.randn(5, k)

    # pseudoinverse lift z@R.T recovers the rank-k projection of h
    h = torch.randn(5, d)
    z_h = h @ R
    h_pinv = z_h @ R.t()
    P = R @ R.t()
    assert torch.allclose(h_pinv, h @ P, atol=1e-5), "z@R.T != projection of h"

    known = KnownRImageDecoder(R, d, k).eval()
    assert known(z).shape == (5, 3, 32, 32)
    oracle = OracleActivationAttacker(R, d, k).eval()
    assert oracle(z).shape == (5, d)
    fuse = GradFusionImageDecoder(k, d, R=R, fuse_pinv=True).eval()
    out = fuse(z, g)
    assert out.shape == (5, 3, 32, 32) and torch.isfinite(out).all()
    # gradient channel actually contributes (output changes when g is zeroed)
    out0 = fuse(z, torch.zeros_like(g))
    assert not torch.allclose(out, out0), "grad fusion ignores g"
    print("  [ok] test_known_r_attack_surface: pinv/oracle/known-R/grad-fusion accept block R")


def main():
    torch.manual_seed(SEED)
    tests = [
        test_exact_vs_unfold,
        test_orthonormal_columns,
        test_block_locality_dial,
        test_payload_constant_across_blocks,
        test_splitmodel_integration,
        test_known_r_attack_surface,
    ]
    for t in tests:
        t()
    print("\nALL BLOCK_RANDOM TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
