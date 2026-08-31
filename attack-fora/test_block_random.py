"""Unit tests for FORA's `block_random` block-local random-projection defense.

Standalone, CPU-only, deterministic, no network. Run under the LightSplit env:

    conda activate LightSplit && cd FORA/FORA && python test_block_random.py

Exits 0 if all tests pass, 1 otherwise, so it can gate a commit. Guards: (a) the
assembled R is a valid orthonormal block-structured projection, (b) the block-size
locality dial behaves (b=1 per-position .. b=16 global), (c) the payload budget is
honest, (d) DefenseWrapper integrates it spatial-in/spatial-out with a WCC hook, and
(e) the other defenses still work.
"""
import sys
import traceback

import torch

from defenses import DefenseWrapper, BlockRandomBottleneck

torch.manual_seed(0)

# FORA cut at MobileNet layer 2: (C, H, W) = (64, 16, 16) -> flat_dim = 16384.
C, H, W = 64, 16, 16
FLAT = C * H * W

# (block, k) -> expected payload_floats = n_blocks * max(1, k // n_blocks).
# block=1 has n_blocks=256, so it pins payload at 256 and OVER-SPENDS when k<256.
EXPECTED_PAYLOAD = {
    (1, 256): 256, (2, 256): 256, (4, 256): 256, (8, 256): 256, (16, 256): 256,
    (2, 128): 128, (4, 128): 128, (1, 128): 256,   # b1@cr16 over-spends
    (2, 64): 64,   (4, 64): 64,   (1, 64): 256,     # b1@cr32 over-spends
}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def bn(block, k):
    return BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W, block=block, k=k)


def test_payload_budget():
    """payload_floats matches n_blocks * out_per_block; over-spend cells are explicit."""
    for (block, k), pay in EXPECTED_PAYLOAD.items():
        m = bn(block, k)
        check(m.payload_floats == pay,
              f"block={block} k={k}: payload {m.payload_floats} != expected {pay}")
        check(m.R.shape == (FLAT, pay),
              f"block={block} k={k}: R shape {tuple(m.R.shape)} != ({FLAT},{pay})")


def test_orthonormal_and_projector():
    """R has orthonormal columns (R.T@R=I), which makes P=R@R.T an idempotent
    projector: P^2 = R(R.T R)R.T = R I R.T = P. We verify the cheap (payload x payload)
    gram rather than materializing the (16384 x 16384) P."""
    for block in (1, 2, 4, 8, 16):
        m = bn(block, 256)
        gram = m.R.T @ m.R
        eye = torch.eye(m.payload_floats)
        check(torch.allclose(gram, eye, atol=1e-5),
              f"block={block}: R.T@R != I (max dev {(gram-eye).abs().max():.2e})")


def test_block_locality_dial():
    """block=1 -> every column's nonzero rows share one spatial position;
    block=16 -> a column spans many spatial positions (global mixing)."""
    # block=1: each tile is one spatial position projecting its 64 channels.
    m1 = bn(1, 256)
    for col in (0, 100, 255):
        rows = torch.nonzero(m1.R[:, col]).flatten()
        spatial = (rows % (H * W)).unique()      # flat row = c*H*W + (y*W+x)
        check(spatial.numel() == 1,
              f"block=1 col {col} touches {spatial.numel()} spatial positions, want 1")
    # block=16: single global tile -> a column touches many spatial positions.
    m16 = bn(16, 256)
    rows = torch.nonzero(m16.R[:, 0]).flatten()
    spatial = (rows % (H * W)).unique()
    check(spatial.numel() > 1,
          f"block=16 col 0 touches {spatial.numel()} spatial positions, want many (global)")


def test_global_equals_fixed_random_shape():
    """block=16 (==H) collapses to a single global dense orthonormal R: one tile,
    full-receptive-field projection of full column rank (structurally fixed_random)."""
    m = bn(16, 256)
    check(m.n_blocks == 1, f"block=16 should be 1 global block, got {m.n_blocks}")
    check(m.in_per_block == FLAT, f"block=16 in_per_block {m.in_per_block} != {FLAT}")
    # Full column rank == orthonormal columns (R.T@R=I, a cheap 256x256), so the
    # rank-256 global projector exists without materializing the 16384x16384 P.
    gram = m.R.T @ m.R
    check(torch.allclose(gram, torch.eye(256), atol=1e-5),
          "block=16 R columns not orthonormal (global projector not full-rank)")


def test_wrapper_roundtrip_and_wcc_hook():
    """DefenseWrapper is spatial-in/spatial-out and exposes bottleneck_z (B,payload)."""
    k, block = 256, 2
    w = DefenseWrapper("block_random", in_channels=C, H=H, W=W, k=k, block=block)
    x = torch.randn(4, C, H, W)
    y = w(x)
    check(y.shape == x.shape, f"wrapper must preserve spatial shape, got {tuple(y.shape)}")
    check("bottleneck_z" in w.last_info, "block_random must expose bottleneck_z for WCC")
    z = w.last_info["bottleneck_z"]
    check(z.shape == (4, w.module.payload_floats),
          f"bottleneck_z shape {tuple(z.shape)} != (4,{w.module.payload_floats})")


def test_straight_through_gradient():
    """An upstream conv feeding the defense receives nonzero gradient."""
    conv = torch.nn.Conv2d(C, C, 3, padding=1)
    w = DefenseWrapper("block_random", in_channels=C, H=H, W=W, k=256, block=4)
    x = torch.randn(4, C, H, W)
    w(conv(x)).sum().backward()
    gnorm = sum(p.grad.abs().sum().item() for p in conv.parameters() if p.grad is not None)
    check(gnorm > 0, "upstream conv received zero gradient (graph broken)")


def test_divisibility_guard():
    """A block that does not divide H must raise (3 does not divide 16)."""
    raised = False
    try:
        bn(3, 256)
    except ValueError:
        raised = True
    check(raised, "block=3 must raise (does not divide 16)")


def test_no_regression():
    """Pre-existing defenses still construct and forward a dummy spatial batch."""
    x = torch.randn(8, C, H, W)
    for dtype, k in (("none", 32), ("learned_1x1", 16),
                     ("fixed_random", 256), ("topk_sparse", 256)):
        w = DefenseWrapper(dtype, in_channels=C, H=H, W=W, k=k)
        y = w(x)
        check(y.shape == x.shape, f"{dtype}: output shape {tuple(y.shape)} != {tuple(x.shape)}")
        check(bool(torch.isfinite(y).all()), f"{dtype}: non-finite output after adding block_random")


TESTS = [
    ("payload_budget", test_payload_budget),
    ("orthonormal_and_projector", test_orthonormal_and_projector),
    ("block_locality_dial", test_block_locality_dial),
    ("global_equals_fixed_random_shape", test_global_equals_fixed_random_shape),
    ("wrapper_roundtrip_and_wcc_hook", test_wrapper_roundtrip_and_wcc_hook),
    ("straight_through_gradient", test_straight_through_gradient),
    ("divisibility_guard", test_divisibility_guard),
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
