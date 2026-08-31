"""Unit tests for the `topk_sparse` matched-byte sparsification defense in unsplit.

Standalone, CPU-only, deterministic, no network. Run from the unsplit/ dir:

    cd unsplit && python test_topk_sparse.py

Exits 0 if all tests pass, 1 otherwise. Guards the unsplit port of the
the utility harness `topk_sparse` baseline:
  (a) the bottleneck keeps exactly k_s top-magnitude coords with identity lift,
  (b) victim/clone payload shapes match and build_target_mask flags zero columns,
  (c) the byte-matched k_s lands at (4k)//6, and
  (d) the existing defenses still construct and forward.
"""
import sys
import traceback

import torch

from unsplit.bottlenecks import TopKSparseBottleneck, VALUE_BYTES, INDEX_BYTES
from unsplit.defended_models import DefendedSplitModel, build_bottleneck
from unsplit.defended_clone import DefendedClone, build_target_mask
from unsplit.models import CifarNet

torch.manual_seed(0)

# cifar split=4 -> (64,16,16); activation_dim = 16384.
IN_CH, HW = 64, (16, 16)
ACT_DIM = IN_CH * HW[0] * HW[1]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_sparsity_and_identity_lift():
    """Keeps exactly k_s top-|h| coords/row, values intact, lift_back is identity."""
    k_s = 341
    bn = TopKSparseBottleneck(k_s=k_s)
    h = torch.randn(8, 4096)
    z, info = bn(h)
    check(info == {}, f"topk_sparse must emit empty info (got {list(info)})")
    check(z.shape == h.shape, f"payload must stay dense d-dim, got {tuple(z.shape)}")
    nnz = (z != 0).sum(dim=1)
    check(bool((nnz == k_s).all()), f"each row must keep {k_s} nonzeros, got {nnz.tolist()}")
    _, idx_exp = torch.topk(h.abs(), k_s, dim=1)
    mask_exp = torch.zeros_like(h).scatter(1, idx_exp, 1.0)
    check(bool((mask_exp.bool() == (z != 0)).all()), "kept positions are not the top-k_s by |h|")
    check(torch.equal(z, h * mask_exp), "kept values must equal the original activations")
    check(torch.equal(bn.lift_back(z, info), z), "lift_back must be identity")


def test_matched_byte_k_s():
    """build_bottleneck sizes k_s = (4k)//6 so 6*k_s bytes <= 4*k bytes, within one entry."""
    for k in (2048, 1024, 512):
        bn = build_bottleneck("topk_sparse", k=k, in_channels=IN_CH, spatial_hw=HW)
        ks_exp = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))
        check(bn.k_s == ks_exp, f"k={k}: k_s {bn.k_s} != {ks_exp}")
        ls_bytes = k * VALUE_BYTES
        sparse_bytes = bn.k_s * (VALUE_BYTES + INDEX_BYTES)
        check(sparse_bytes <= ls_bytes, f"k={k}: {sparse_bytes}B exceeds dense {ls_bytes}B")
        check(ls_bytes - sparse_bytes < (VALUE_BYTES + INDEX_BYTES),
              f"k={k}: under-spends dense budget by more than one entry")


def test_victim_clone_shapes_and_mask():
    """Victim and clone payloads match at (B, activation_dim); mask flags zero columns."""
    B, k = 2, 2048
    x = torch.randn(B, 3, 32, 32)
    victim = DefendedSplitModel(CifarNet, split_layer=4, defense="topk_sparse", k=k,
                                in_channels=IN_CH, spatial_hw=HW)
    clone = DefendedClone(CifarNet, split_layer=4, defense="topk_sparse", k=k,
                          in_channels=IN_CH, spatial_hw=HW)
    vp = victim.compress_payload(x)
    cp = clone(x, end=4)
    check(vp.shape == (B, ACT_DIM), f"victim payload {tuple(vp.shape)} != (B,{ACT_DIM})")
    check(vp.shape == cp.shape, f"shape mismatch victim={tuple(vp.shape)} clone={tuple(cp.shape)}")
    # clone is dense (Identity compress); victim is sparse.
    check(int((cp != 0).sum()) > int((vp != 0).sum()), "clone should be denser than sparse victim")
    # single-sample target: mask must flag the zeroed (non-kept) columns.
    mask = build_target_mask(vp[:1])
    check(mask is not None, "build_target_mask returned None on a sparse payload")
    check(mask.sum() < mask.numel(), f"mask should zero some cols ({mask.sum()}/{mask.numel()})")


def test_full_forward_and_grad():
    """Defended model forwards finitely and the client head gets nonzero gradient."""
    model = DefendedSplitModel(CifarNet, split_layer=4, defense="topk_sparse", k=2048,
                               in_channels=IN_CH, spatial_hw=HW)
    model.train()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    logits, info = model.full_forward(x)
    check(tuple(logits.shape) == (8, 10), f"logits {tuple(logits.shape)} != (8,10)")
    check(bool(torch.isfinite(logits).all()), "non-finite logits")
    check("bottleneck_z" not in info, "topk_sparse must not emit bottleneck_z (no WCC)")
    torch.nn.functional.cross_entropy(logits, y).backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.client.parameters() if p.grad is not None)
    check(gnorm > 0, "client head got zero gradient (straight-through path broken)")


def test_no_regression():
    """Pre-existing defenses still construct and forward."""
    x = torch.randn(2, 3, 32, 32)
    for defense in ("learned_1x1", "fixed_random"):
        m = DefendedSplitModel(CifarNet, split_layer=4, defense=defense, k=2048,
                               in_channels=IN_CH, spatial_hw=HW)
        logits, _ = m.full_forward(x)
        check(tuple(logits.shape) == (2, 10), f"{defense}: logits {tuple(logits.shape)}")
        check(bool(torch.isfinite(logits).all()), f"{defense}: non-finite logits")


TESTS = [
    ("sparsity_and_identity_lift", test_sparsity_and_identity_lift),
    ("matched_byte_k_s", test_matched_byte_k_s),
    ("victim_clone_shapes_and_mask", test_victim_clone_shapes_and_mask),
    ("full_forward_and_grad", test_full_forward_and_grad),
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
