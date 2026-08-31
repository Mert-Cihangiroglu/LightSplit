"""Unit tests for the `block_random` block-local random-projection defense in unsplit.

Standalone, CPU-only, deterministic, no network. Run from the unsplit/ dir:

    cd unsplit && python test_block_random.py

Exits 0 if all tests pass, 1 otherwise. Guards the unsplit port of the utility harness
`block_random` defense (the privacy/utility locality dial):
  (a) the bottleneck tiles + projects to (B, n_blocks, out_per_block) and lifts back,
  (b) R is an orthonormal client-only buffer,
  (c) the clone's stride-block Conv2d is EXACTLY the victim's shared-R per-tile
      projection (the alignment guard that locks the reshape/ordering forever),
  (d) block=full reduces to a single global projection (== fixed_random),
  (e) block=2 is byte-matched (payload_floats == k) at the grid CRs, and
  (f) adding it does not break the existing defenses.
"""
import sys
import traceback

import torch
import torch.nn as nn

from unsplit.bottlenecks import BlockRandomBottleneck
from unsplit.defended_models import DefendedSplitModel, build_bottleneck
from unsplit.defended_clone import DefendedClone, build_target_mask
from unsplit.models import CifarNet

torch.manual_seed(0)

# cifar split=4 -> (64,16,16).
IN_CH, HW = 64, (16, 16)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_block_projection_and_lift():
    """Payload is (B, n_blocks, out_per_block); lift round-trips to (B,C,H,W);
    R is an orthonormal buffer (not a trainable parameter)."""
    B, C, H, W, block, k = 2, 8, 16, 16, 2, 256
    bn = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W, block=block, k=k)
    z = torch.randn(B, C, H, W)
    payload, info = bn(z)
    n_blocks = (H // block) * (W // block)
    out_per_block = max(1, k // n_blocks)
    check(payload.shape == (B, n_blocks, out_per_block),
          f"payload {tuple(payload.shape)} != {(B, n_blocks, out_per_block)}")
    check(bn.lift_back(payload, info).shape == (B, C, H, W), "lift_back must give (B,C,H,W)")
    check("bottleneck_z" in info and info["bottleneck_z"].shape == (B, n_blocks * out_per_block),
          f"bottleneck_z missing/wrong: {info.keys()}")
    # R orthonormal columns: R^T R = I
    eye = bn.R.T @ bn.R
    check(torch.allclose(eye, torch.eye(bn.R.shape[1]), atol=1e-5), "R columns not orthonormal")
    # R is a buffer, not a parameter (never trained, never sent).
    param_ids = {id(p) for p in bn.parameters()}
    check(id(bn.R) not in param_ids, "R must be a buffer, not a trainable parameter")
    check(any(name == "R" for name, _ in bn.named_buffers()), "R must be a registered buffer")


def test_conv_equals_block_projection():
    """ALIGNMENT GUARD: a stride-block Conv2d whose weight is R.T reshaped reproduces the
    victim's _to_blocks(z) @ R bit-for-bit (proving the (B,n_blocks,out_per_block)
    reshape/transpose and block ordering match)."""
    B, C, H, W, block, k = 3, 8, 16, 16, 2, 256
    bn = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W, block=block, k=k)
    out_per_block = bn.out_per_block
    conv = nn.Conv2d(C, out_per_block, kernel_size=block, stride=block)
    with torch.no_grad():
        conv.weight.copy_(bn.R.T.reshape(out_per_block, C, block, block))
        conv.bias.zero_()
    z = torch.randn(B, C, H, W)
    clone_payload = conv(z).flatten(2).transpose(1, 2)   # (B, n_blocks, out_per_block)
    victim_payload, _ = bn(z)
    diff = (clone_payload - victim_payload).abs().max().item()
    check(diff < 1e-5, f"conv != block projection (max abs diff {diff:.2e})")


def test_block_full_equals_global_projection():
    """block=full (single tile) reduces to one global dense R on the flattened
    activation -- i.e. exactly the FixedRandomBottleneck operation."""
    B, C, H, W, k = 2, 8, 12, 12, 36
    bn = BlockRandomBottleneck(channels=C, hw_h=H, hw_w=W, block=H, k=k)  # H==W
    check(bn.n_blocks == 1, f"block=full should give n_blocks=1, got {bn.n_blocks}")
    z = torch.randn(B, C, H, W)
    payload = bn(z)[0].squeeze(1)             # (B, k)
    global_proj = z.flatten(1) @ bn.R         # what fixed_random computes
    check(torch.allclose(payload, global_proj, atol=1e-5),
          "block=full is not a single global projection")


def test_victim_clone_shapes_and_mask():
    """Victim and clone payloads match at (B, n_blocks, out_per_block); the 3D dense
    payload yields no target mask."""
    B, k, block = 2, 2048, 2
    x = torch.randn(B, 3, 32, 32)
    victim = DefendedSplitModel(CifarNet, split_layer=4, defense="block_random", k=k,
                                in_channels=IN_CH, spatial_hw=HW, block_size=block)
    clone = DefendedClone(CifarNet, split_layer=4, defense="block_random", k=k,
                          in_channels=IN_CH, spatial_hw=HW, block_size=block)
    vp = victim.compress_payload(x)
    cp = clone(x, end=4)
    n_blocks = (HW[0] // block) * (HW[1] // block)
    out_per_block = max(1, k // n_blocks)
    check(vp.shape == (B, n_blocks, out_per_block), f"victim payload {tuple(vp.shape)}")
    check(vp.shape == cp.shape, f"shape mismatch victim={tuple(vp.shape)} clone={tuple(cp.shape)}")
    check(build_target_mask(vp[:1]) is None, "3D block payload must not produce a target mask")


def test_payload_floats_matches_k():
    """block=2 is byte-matched (payload_floats == k) at all grid (split, CR) cells."""
    # (in_channels, H, W, k) for the canonical grid at block=2.
    cases = [
        (16, 8, 8, 128), (16, 8, 8, 64), (16, 8, 8, 32),       # mnist/f_mnist split=4
        (8, 12, 12, 144), (8, 12, 12, 72), (8, 12, 12, 36),    # mnist/f_mnist split=2
    ]
    for C, H, W, k in cases:
        bn = build_bottleneck("block_random", k=k, in_channels=C, spatial_hw=(H, W), block_size=2)
        check(bn.payload_floats == k,
              f"C{C} {H}x{W} k={k}: payload_floats {bn.payload_floats} != k")


def test_full_forward_and_grad():
    """Defended model forwards finitely, client gets nonzero gradient, emits bottleneck_z."""
    model = DefendedSplitModel(CifarNet, split_layer=4, defense="block_random", k=2048,
                               in_channels=IN_CH, spatial_hw=HW, block_size=2)
    model.train()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    logits, info = model.full_forward(x)
    check(tuple(logits.shape) == (8, 10), f"logits {tuple(logits.shape)} != (8,10)")
    check(bool(torch.isfinite(logits).all()), "non-finite logits")
    check("bottleneck_z" in info, "block_random must emit bottleneck_z (for WCC)")
    torch.nn.functional.cross_entropy(logits, y).backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.client.parameters() if p.grad is not None)
    check(gnorm > 0, "client head got zero gradient (path broken)")


def test_divisibility_guard():
    """A block that does not divide the spatial dims raises a clear error."""
    raised = False
    try:
        BlockRandomBottleneck(channels=8, hw_h=12, hw_w=12, block=5, k=36)
    except ValueError:
        raised = True
    check(raised, "block not dividing spatial dims must raise ValueError")


def test_no_regression():
    """Pre-existing defenses still construct and forward."""
    x = torch.randn(2, 3, 32, 32)
    for defense in ("learned_1x1", "fixed_random", "topk_sparse"):
        m = DefendedSplitModel(CifarNet, split_layer=4, defense=defense, k=2048,
                               in_channels=IN_CH, spatial_hw=HW)
        logits, _ = m.full_forward(x)
        check(tuple(logits.shape) == (2, 10), f"{defense}: logits {tuple(logits.shape)}")
        check(bool(torch.isfinite(logits).all()), f"{defense}: non-finite logits")


TESTS = [
    ("block_projection_and_lift", test_block_projection_and_lift),
    ("conv_equals_block_projection", test_conv_equals_block_projection),
    ("block_full_equals_global_projection", test_block_full_equals_global_projection),
    ("victim_clone_shapes_and_mask", test_victim_clone_shapes_and_mask),
    ("payload_floats_matches_k", test_payload_floats_matches_k),
    ("full_forward_and_grad", test_full_forward_and_grad),
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
