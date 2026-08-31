"""Unit tests for FORA's `topk_sparse` matched-byte sparsification defense.

Standalone, CPU-only, deterministic, no network. Run under the LightSplit env:

    conda activate LightSplit && cd FORA/FORA && python test_topk_sparse.py

Exits 0 if all tests pass, 1 otherwise, so it can gate a commit. It guards the two
promises of the change: (a) topk_sparse is matched-byte against the LightSplit
budget k, and (b) adding it does not break the existing DefenseWrapper modes.
"""
import sys
import traceback

import torch

from defenses import (DefenseWrapper, TopKSparseBottleneck,
                      VALUE_BYTES, INDEX_BYTES)

torch.manual_seed(0)

# FORA cut at MobileNet layer 2: (C, H, W) = (64, 16, 16) -> flat_dim = 16384.
C, H, W = 64, 16, 16
FLAT = C * H * W

# k (LightSplit budget) -> expected k_s = 4k//6. Stated independently of the impl
# formula so a formula change is caught.
EXPECTED_KS = {32: 21, 128: 85, 256: 170, 512: 341}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_matched_byte_k_s():
    """DefenseWrapper computes k_s = 4k//6, and that is <= LightSplit bytes, within one entry."""
    for k, ks_exp in EXPECTED_KS.items():
        w = DefenseWrapper("topk_sparse", in_channels=C, H=H, W=W, k=k)
        check(w.k_s == ks_exp, f"k={k}: k_s {w.k_s} != expected {ks_exp}")
        ls_bytes = k * VALUE_BYTES
        ts_bytes = w.k_s * (VALUE_BYTES + INDEX_BYTES)
        check(ts_bytes <= ls_bytes,
              f"k={k}: topk_sparse {ts_bytes} B exceeds LightSplit {ls_bytes} B")
        check(ls_bytes - ts_bytes < (VALUE_BYTES + INDEX_BYTES),
              f"k={k}: under-spends LightSplit {ls_bytes} B by more than one entry")


def test_sparsity_and_correctness():
    """Keeps exactly k_s nonzero coords/row at the top-k_s |h| positions, values intact."""
    k_s = 170
    bn = TopKSparseBottleneck(k_s=k_s)
    h = torch.randn(8, FLAT)
    z, info = bn(h)
    check(info == {}, f"topk_sparse must emit empty info (got keys {list(info)})")
    check(z.shape == h.shape, f"payload must stay dense d-dim, got {tuple(z.shape)}")
    nnz = (z != 0).sum(dim=1)
    check(bool((nnz == k_s).all()),
          f"each row must keep exactly {k_s} nonzeros, got {nnz.tolist()}")
    _, idx_exp = torch.topk(h.abs(), k_s, dim=1)
    mask_exp = torch.zeros_like(h).scatter(1, idx_exp, 1.0)
    check(bool((mask_exp.bool() == (z != 0)).all()),
          "kept positions are not the top-k_s by |h|")
    check(torch.equal(z, h * mask_exp), "kept values must equal the original activations")
    check(torch.equal(bn.lift_back(z, info), z), "lift_back must be identity")


def test_wrapper_spatial_roundtrip():
    """Wrapper is spatial-in / spatial-out and keeps k_s nonzeros across the flat map."""
    k = 256
    w = DefenseWrapper("topk_sparse", in_channels=C, H=H, W=W, k=k)
    x = torch.randn(4, C, H, W)
    y = w(x)
    check(y.shape == x.shape, f"wrapper must preserve spatial shape, got {tuple(y.shape)}")
    check(w.last_info == {}, f"topk_sparse must leave last_info empty (got {list(w.last_info)})")
    nnz = (y.reshape(4, FLAT) != 0).sum(dim=1)
    check(bool((nnz == w.k_s).all()),
          f"each sample must keep exactly {w.k_s} nonzeros, got {nnz.tolist()}")


def test_straight_through_gradient():
    """A conv feeding the defense receives nonzero gradient (straight-through path)."""
    k = 256
    conv = torch.nn.Conv2d(C, C, 3, padding=1)
    w = DefenseWrapper("topk_sparse", in_channels=C, H=H, W=W, k=k)
    x = torch.randn(4, C, H, W)
    out = w(conv(x))
    out.sum().backward()
    gnorm = sum(p.grad.abs().sum().item() for p in conv.parameters() if p.grad is not None)
    check(gnorm > 0, "upstream conv received zero gradient (straight-through path broken)")


def test_no_regression():
    """Pre-existing defenses still construct and forward a dummy spatial batch."""
    x = torch.randn(8, C, H, W)
    for dtype, k in (("none", 32), ("learned_1x1", 16), ("fixed_random", 256)):
        w = DefenseWrapper(dtype, in_channels=C, H=H, W=W, k=k)
        y = w(x)
        check(y.shape == x.shape, f"{dtype}: output shape {tuple(y.shape)} != {tuple(x.shape)}")
        check(bool(torch.isfinite(y).all()), f"{dtype}: non-finite output after adding topk_sparse")


TESTS = [
    ("matched_byte_k_s", test_matched_byte_k_s),
    ("sparsity_and_correctness", test_sparsity_and_correctness),
    ("wrapper_spatial_roundtrip", test_wrapper_spatial_roundtrip),
    ("straight_through_gradient", test_straight_through_gradient),
    ("no_regression", test_no_regression),
]


def main():
    failures = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print("-" * 60)
    if failures:
        print(f"{failures}/{len(TESTS)} test(s) FAILED")
        return 1
    print(f"ALL {len(TESTS)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
