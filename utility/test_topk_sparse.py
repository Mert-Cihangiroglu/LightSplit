"""Unit tests for the `topk_sparse` matched-byte sparsification baseline.

Standalone, CPU-only, deterministic, no network (backbones build from scratch).
Run under the LightSplit conda env:

    conda activate LightSplit && cd the utility harness && python test_topk_sparse.py

Exits 0 if all tests pass, 1 otherwise, so it can gate a commit:

    python test_topk_sparse.py && git commit ...

It guards the two promises of the change:
  (a) topk_sparse is matched-byte against LightSplit at each CR, and
  (b) adding it does not break the existing modes.
"""
import sys
import traceback

import torch
import torch.nn.functional as F

from train_split import SplitModel, TopKSparseBottleneck, VALUE_BYTES, INDEX_BYTES

torch.manual_seed(0)

# (dataset, CR) -> (activation_dim d, k=d//CR, expected k_s, expected payload_dim).
# Expected numbers are stated independently (not recomputed from the impl formula)
# so a formula change is caught. Source: the verified matched-byte table in the plan.
EXPECTED = {
    ("cifar10", 8):        (4096, 512, 341, 512),
    ("cifar10", 16):       (4096, 256, 170, 255),
    ("cifar10", 32):       (4096, 128, 85, 128),
    ("fmnist", 8):         (2880, 360, 240, 360),
    ("fmnist", 16):        (2880, 180, 120, 180),
    ("fmnist", 32):        (2880, 90, 60, 90),
}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def build(dataset, mode, k):
    return SplitModel(dataset=dataset, mode=mode, k=k)


def test_matched_byte_payload():
    """_k_s and payload_dim match the verified table; CR lands on nominal value."""
    for (dataset, cr), (d, k, ks_exp, pay_exp) in EXPECTED.items():
        m = build(dataset, "topk_sparse", k)
        check(m.activation_dim == d,
              f"{dataset}: activation_dim {m.activation_dim} != {d}")
        check(m.k == k, f"{dataset} CR{cr}: model.k {m.k} != {k}")
        check(m._k_s == ks_exp,
              f"{dataset} CR{cr}: _k_s {m._k_s} != expected {ks_exp}")
        check(m.payload_dim == pay_exp,
              f"{dataset} CR{cr}: payload_dim {m.payload_dim} != expected {pay_exp}")
        cr_eff = d / m.payload_dim
        check(abs(cr_eff - cr) < 0.2,
              f"{dataset} CR{cr}: effective CR {cr_eff:.2f} drifts from nominal {cr}")


def test_sparsity_and_correctness():
    """Keeps exactly k_s nonzero coords/row, at the top-k_s |h| positions, values intact."""
    k_s = 341
    bn = TopKSparseBottleneck(k_s=k_s)
    h = torch.randn(8, 4096)
    z, info = bn(h)
    check(info == {}, f"topk_sparse must emit empty info (got keys {list(info)})")
    check(z.shape == h.shape, f"payload must stay dense d-dim, got {tuple(z.shape)}")
    nnz = (z != 0).sum(dim=1)
    check(bool((nnz == k_s).all()),
          f"each row must keep exactly {k_s} nonzeros, got {nnz.tolist()}")
    # the kept positions must be exactly the top-k_s by magnitude
    _, idx_exp = torch.topk(h.abs(), k_s, dim=1)
    mask_exp = torch.zeros_like(h).scatter(1, idx_exp, 1.0)
    check(bool((mask_exp.bool() == (z != 0)).all()),
          "kept positions are not the top-k_s by |h|")
    # values at kept positions are the originals (zeros elsewhere)
    check(torch.equal(z, h * mask_exp),
          "kept values must equal the original activations")


def test_trains_straight_through():
    """Full cifar10 model does a finite forward + nonzero head gradient (trainable)."""
    m = build("cifar10", "topk_sparse", 512)
    m.train()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    logits, _ = m(x)
    check(tuple(logits.shape) == (8, 10), f"logits shape {tuple(logits.shape)} != (8, 10)")
    check(bool(torch.isfinite(logits).all()), "logits contain non-finite values")
    loss = F.cross_entropy(logits, y)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item()
                    for p in m.head.parameters() if p.grad is not None)
    check(grad_norm > 0,
          "client head received zero gradient (straight-through path broken)")


def test_matched_byte_vs_lightsplit():
    """At each CR, topk_sparse bytes <= LightSplit bytes and within one 6-byte entry."""
    for (dataset, cr), (d, k, ks_exp, pay_exp) in EXPECTED.items():
        m = build(dataset, "topk_sparse", k)
        ls_bytes = k * VALUE_BYTES
        topk_bytes = m._k_s * (VALUE_BYTES + INDEX_BYTES)
        check(topk_bytes <= ls_bytes,
              f"{dataset} CR{cr}: topk {topk_bytes} B exceeds LightSplit {ls_bytes} B")
        check(ls_bytes - topk_bytes < (VALUE_BYTES + INDEX_BYTES),
              f"{dataset} CR{cr}: topk {topk_bytes} B under-spends LightSplit "
              f"{ls_bytes} B by more than one entry")


def test_no_regression():
    """Pre-existing modes still construct and forward a dummy cifar10 batch."""
    x = torch.randn(8, 3, 32, 32)
    for mode in ("fixed_random", "fixed_random_ll"):
        m = build("cifar10", mode, 512)
        m.train()
        logits, _ = m(x)
        check(tuple(logits.shape) == (8, 10),
              f"mode={mode}: logits shape {tuple(logits.shape)} != (8, 10)")
        check(bool(torch.isfinite(logits).all()),
              f"mode={mode}: non-finite logits after adding topk_sparse")


TESTS = [
    ("matched_byte_payload", test_matched_byte_payload),
    ("sparsity_and_correctness", test_sparsity_and_correctness),
    ("trains_straight_through", test_trains_straight_through),
    ("matched_byte_vs_lightsplit", test_matched_byte_vs_lightsplit),
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
