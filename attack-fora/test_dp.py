"""Unit tests for the DP (per-sample L2 clip + Gaussian noise) path in DefenseWrapper.

Standalone, CPU-only, deterministic, no network. Run under the LightSplit env:

    conda activate LightSplit && cd FORA/FORA && python test_dp.py

Exits 0 if all tests pass, 1 otherwise, so it can gate a commit. Guards: (a) DP is
INERT when clip_norm=0 -> the un-noised paths (raw + block_random) stay byte-identical
(the non-damaging guarantee), (b) clipping bounds the per-sample payload norm and records
the clip fraction, (c) the noise has the calibrated std (sigma * clip_norm), (d) gradients
flow through the DP path, (e) raw+DP / block_random+DP preserve the (B,64,16,16) shape and
DP is actually active, and (f) the other defenses still forward with DP off.
"""
import sys
import traceback

import torch

from defenses import DefenseWrapper

torch.manual_seed(0)

C, H, W = 64, 16, 16
FLAT = C * H * W


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def wrap(defense, clip_norm=0.0, noise=0.0, k=256, block=2):
    return DefenseWrapper(defense, in_channels=C, H=H, W=W, k=k, block=block,
                          clip_norm=clip_norm, noise_multiplier=noise)


def test_clip0_byte_identical_raw():
    """none/raw with clip_norm=0 returns the activation untouched, last_info empty."""
    w = wrap("none", clip_norm=0.0)
    h = torch.randn(4, C, H, W)
    out = w(h)
    check(torch.equal(out, h), "raw clip0 changed the activation")
    check(w.last_info == {}, "raw clip0 should leave last_info empty")


def test_clip0_byte_identical_block_random():
    """block_random with clip_norm=0 equals the no-DP flatten->compress->lift path exactly."""
    torch.manual_seed(1)
    w = wrap("block_random", clip_norm=0.0, k=256, block=2)
    h = torch.randn(4, C, H, W)
    out = w(h)
    # manual no-DP reference through the very same module (R is fixed on the instance)
    B = h.size(0)
    payload, info = w.module(h.reshape(B, w.flat_dim))
    ref = w.module.lift_back(payload, info).view(B, C, H, W)
    check(torch.equal(out, ref), "block_random clip0 not byte-identical to no-DP path")


def test_clip_bounds_norm():
    """clip (noise off) bounds every per-sample payload norm to <= C; below-C rows untouched."""
    Cn = 3.0
    w = wrap("none", clip_norm=Cn, noise=0.0)
    z = torch.randn(200, 256)
    z = z / z.norm(dim=1, keepdim=True)               # unit rows
    z[:100] *= 2 * Cn                                 # 100 rows above the clip
    z[100:] *= 0.5 * Cn                               # 100 rows below the clip
    info = {}
    out = w._apply_dp(z.clone(), info)
    norms = out.norm(dim=1)
    check(bool((norms <= Cn + 1e-4).all()), f"clipped norm exceeds C: max {norms.max():.4f}")
    check(torch.allclose(out[100:], z[100:], atol=1e-5), "below-C rows were modified")
    check(abs(info["dp_clip_fraction"].item() - 0.5) < 1e-6,
          f"clip_fraction {info['dp_clip_fraction'].item()} != 0.5")


def test_noise_std():
    """With no clipping, the added noise has std ~ sigma * clip_norm per element."""
    torch.manual_seed(2)
    Cn, sigma = 10.0, 0.5
    w = wrap("none", clip_norm=Cn, noise=sigma)
    z = torch.randn(2000, 256) * 0.05                 # row norms ~0.8 << C -> no clipping
    info = {}
    out = w._apply_dp(z.clone(), info)
    check(info["dp_clip_fraction"].item() == 0.0, "unexpected clipping in the no-clip regime")
    std = (out - z).std().item()
    expected = sigma * Cn
    check(abs(std - expected) / expected < 0.05,
          f"noise std {std:.3f} != expected {expected:.3f}")


def test_gradient_flows():
    """Clipping is differentiable: gradient reaches the input through the DP path."""
    w = wrap("block_random", clip_norm=5.0, noise=0.02, k=256, block=2)
    h = torch.randn(4, C, H, W, requires_grad=True)
    w(h).sum().backward()
    check(h.grad is not None, "no gradient reached the input")
    check(bool(torch.isfinite(h.grad).all()), "non-finite gradient through DP")
    check(h.grad.abs().sum().item() > 0, "zero gradient through DP")


def test_raw_dp_active():
    """raw+DP preserves shape, actually changes the activation, and records clip fraction."""
    torch.manual_seed(3)
    w = wrap("none", clip_norm=5.0, noise=0.02)
    h = torch.randn(4, C, H, W)
    out = w(h)
    check(out.shape == h.shape, f"raw+DP shape {tuple(out.shape)} != {(4, C, H, W)}")
    check(not torch.equal(out, h), "raw+DP left the activation unchanged")
    check("dp_clip_fraction" in w.last_info, "raw+DP did not record dp_clip_fraction")


def test_block_random_dp_shape():
    """block_random+DP (incl. block=8) is spatial-in/spatial-out (B,64,16,16)."""
    for block in (1, 2, 4, 8):
        w = wrap("block_random", clip_norm=3.0, noise=0.02, k=256, block=block)
        out = w(torch.randn(2, C, H, W))
        check(out.shape == (2, C, H, W),
              f"block={block}: shape {tuple(out.shape)} != (2,{C},{H},{W})")


def test_other_defenses_off():
    """With DP off (clip_norm=0) the other defenses still forward to the right shape."""
    for defense, k in (("fixed_random", 256),
                       ("learned_1x1", 16), ("topk_sparse", 256)):
        w = wrap(defense, clip_norm=0.0, k=k)
        out = w(torch.randn(8, C, H, W))
        check(out.shape == (8, C, H, W),
              f"{defense}: shape {tuple(out.shape)} != (8,{C},{H},{W})")


TESTS = [
    ("clip0_byte_identical_raw", test_clip0_byte_identical_raw),
    ("clip0_byte_identical_block_random", test_clip0_byte_identical_block_random),
    ("clip_bounds_norm", test_clip_bounds_norm),
    ("noise_std", test_noise_std),
    ("gradient_flows", test_gradient_flows),
    ("raw_dp_active", test_raw_dp_active),
    ("block_random_dp_shape", test_block_random_dp_shape),
    ("other_defenses_off", test_other_defenses_off),
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
