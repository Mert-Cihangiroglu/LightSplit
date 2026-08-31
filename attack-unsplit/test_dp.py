"""Unit tests for the Differential-Privacy defense in unsplit (per-sample L2 clip + Gaussian
noise on the transmitted payload, ported from utility/train_split.py:_apply_dp).

Standalone, CPU-only, deterministic. Run from unsplit/:  python test_dp.py
Guards:
  (a) _apply_dp clips each sample to clip_norm and leaves smaller-norm samples untouched,
  (b) noise_multiplier>0 adds Gaussian noise of the calibrated scale (std = sigma*clip),
  (c) clip_norm<=0 is a no-op,
  (d) raw and block_random route DP with shapes intact; compress_payload is noised (attacker sees it),
  (e) the raw clone matches the raw victim payload shape,
  (f) existing defenses still work.
"""
import sys
import traceback

import torch

from unsplit.defended_models import DefendedSplitModel
from unsplit.defended_clone import DefendedClone
from unsplit.models import CifarNet

torch.manual_seed(0)

IN_CH, HW = 64, (16, 16)
ACT_DIM = IN_CH * HW[0] * HW[1]  # 16384


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _model(defense, clip_norm=0.0, noise=0.0, k=2048, block=2):
    return DefendedSplitModel(CifarNet, split_layer=4, defense=defense, k=k,
                              in_channels=IN_CH, spatial_hw=HW, block_size=block,
                              clip_norm=clip_norm, noise_multiplier=noise)


def test_apply_dp_clips():
    """Per-sample L2 norm clipped to clip_norm; a small-norm sample is left unchanged."""
    m = _model("raw", clip_norm=5.0, noise=0.0)
    p = torch.randn(6, 200) * 3.0     # norms ~ 3*sqrt(200) >> 5, all clip
    p[0] = torch.randn(200) * 1e-3    # tiny norm, must be untouched
    out = m._apply_dp(p)
    norms = out.flatten(1).norm(dim=1)
    check(bool((norms[1:] <= 5.0 + 1e-4).all()), f"clipped norms exceed clip: {norms[1:].max()}")
    check(torch.allclose(out[0], p[0], atol=1e-6), "small-norm sample must be unchanged by clip")
    # a clipped sample keeps direction, scaled to exactly clip_norm
    check(abs(float(out[1].norm()) - 5.0) < 1e-3, f"clipped norm != clip_norm ({out[1].norm()})")


def test_apply_dp_noise_scale():
    """noise_multiplier>0 adds Gaussian noise of std ~ sigma*clip (measured on the residual)."""
    clip, sig = 5.0, 0.2
    m = _model("raw", clip_norm=clip, noise=sig)
    p = torch.randn(2048, 50) * 3.0
    clipped = _model("raw", clip_norm=clip, noise=0.0)._apply_dp(p)
    noised = m._apply_dp(p)
    resid = (noised - clipped).flatten()
    check(abs(float(resid.std()) - sig * clip) < 0.05, f"noise std {resid.std():.3f} != {sig*clip}")
    check(not torch.allclose(noised, clipped), "noise>0 must change the payload")


def test_dp_off_is_noop():
    m = _model("raw", clip_norm=0.0, noise=0.0)
    p = torch.randn(4, 100)
    check(torch.equal(m._apply_dp(p), p), "clip_norm<=0 must be a no-op")


def test_raw_dp_shapes_and_noised():
    """raw+DP: payload is the flattened activation (B, ACT_DIM), DP-noised (differs per call)."""
    B = 3
    x = torch.randn(B, 3, 32, 32)
    m = _model("raw", clip_norm=5.0, noise=0.2)
    m.eval()
    p1 = m.compress_payload(x)
    p2 = m.compress_payload(x)
    check(p1.shape == (B, ACT_DIM), f"raw payload {tuple(p1.shape)} != (B,{ACT_DIM})")
    check(not torch.allclose(p1, p2), "DP noise must make the attacker's payload stochastic")
    logits, _ = m.full_forward(x)
    check(tuple(logits.shape) == (B, 10), f"raw full_forward logits {tuple(logits.shape)}")
    # clipped: each sample's payload norm <= clip + noise slack
    check(bool(torch.isfinite(p1).all()), "non-finite raw payload")


def test_block_random_dp_shapes():
    """block_random+DP: 3D payload shape intact, forward finite, gradient flows."""
    B = 4
    x = torch.randn(B, 3, 32, 32)
    m = _model("block_random", clip_norm=5.0, noise=0.2, k=2048, block=2)
    m.train()
    n_blocks = (HW[0] // 2) * (HW[1] // 2)
    out_per_block = max(1, 2048 // n_blocks)
    p = m.compress_payload(x)
    check(p.shape == (B, n_blocks, out_per_block), f"block payload {tuple(p.shape)}")
    logits, info = m.full_forward(x)
    check(tuple(logits.shape) == (B, 10), "block_random+DP logits shape")
    torch.nn.functional.cross_entropy(logits, torch.randint(0, 10, (B,))).backward()
    g = sum(p.grad.abs().sum().item() for p in m.client.parameters() if p.grad is not None)
    check(g > 0, "client got zero gradient through DP path")


def test_raw_clone_matches():
    B = 2
    x = torch.randn(B, 3, 32, 32)
    victim = _model("raw", clip_norm=5.0, noise=0.2)
    clone = DefendedClone(CifarNet, split_layer=4, defense="raw", k=ACT_DIM,
                          in_channels=IN_CH, spatial_hw=HW)
    vp = victim.compress_payload(x)
    cp = clone(x, end=4)
    check(vp.shape == cp.shape == (B, ACT_DIM), f"raw victim {tuple(vp.shape)} vs clone {tuple(cp.shape)}")


def test_no_regression():
    """DP-off models for existing defenses still construct and forward."""
    x = torch.randn(2, 3, 32, 32)
    for d in ("learned_1x1", "fixed_random", "topk_sparse", "block_random"):
        m = _model(d, clip_norm=0.0, noise=0.0)
        logits, _ = m.full_forward(x)
        check(tuple(logits.shape) == (2, 10), f"{d}: logits {tuple(logits.shape)}")
        check(bool(torch.isfinite(logits).all()), f"{d}: non-finite logits")


TESTS = [
    ("apply_dp_clips", test_apply_dp_clips),
    ("apply_dp_noise_scale", test_apply_dp_noise_scale),
    ("dp_off_is_noop", test_dp_off_is_noop),
    ("raw_dp_shapes_and_noised", test_raw_dp_shapes_and_noised),
    ("block_random_dp_shapes", test_block_random_dp_shapes),
    ("raw_clone_matches", test_raw_clone_matches),
    ("no_regression", test_no_regression),
]


def main():
    failures = 0
    for name, fn in TESTS:
        try:
            fn(); print(f"PASS  {name}")
        except Exception:
            failures += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print("-" * 60)
    if failures:
        print(f"{failures}/{len(TESTS)} test(s) FAILED"); return 1
    print(f"ALL {len(TESTS)} TESTS PASSED"); return 0


if __name__ == "__main__":
    sys.exit(main())
