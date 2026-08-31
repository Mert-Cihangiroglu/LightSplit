"""Smoke tests for the `topk_sparse` matched-byte bottleneck in the FSHA repo.

Standalone, CPU-only, deterministic. Run under the sdar env:

    python test_topk_sparse_tf.py

Exits 0 if all tests pass, 1 otherwise. Guards the two promises of the port:
  (a) topk_sparse is byte-matched (same k_s / payload_dim as the utility harness source), and
  (b) it keeps exactly the top-k_s magnitude coordinates and plugs into FSHA unchanged.
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only, deterministic

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bottlenecks
import FSHA

tf.random.set_seed(0)
np.random.seed(0)

# FSHA cut (id-setup 3 -> level 4) is 4x4x256 = 4096, matching the source cifar10 rows.
CUT_SHAPE = (4, 4, 256)
D = 4096
# CR -> (expected k_s, expected payload_dim). Stated independently so a formula change is caught.
EXPECTED = {8: (341, 512), 16: (170, 255), 32: (85, 128)}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_matched_byte_payload():
    """k_s / payload_dim match the verified byte-matched table; effective CR lands on nominal."""
    for cr, (ks_exp, pay_exp) in EXPECTED.items():
        bn = bottlenecks.make_bottleneck("topk_sparse", CUT_SHAPE, compression_ratio=cr)
        check(bn.activation_dim == D, f"CR{cr}: activation_dim {bn.activation_dim} != {D}")
        check(bn.nominal_k == D // cr, f"CR{cr}: nominal_k {bn.nominal_k} != {D // cr}")
        check(bn.k_s == ks_exp, f"CR{cr}: k_s {bn.k_s} != expected {ks_exp}")
        check(bn.payload_dim == pay_exp, f"CR{cr}: payload_dim {bn.payload_dim} != expected {pay_exp}")
        check(bn.k == bn.k_s, f"CR{cr}: k {bn.k} != k_s {bn.k_s}")
        cr_eff = D / bn.payload_dim
        check(abs(cr_eff - cr) < 0.2, f"CR{cr}: effective CR {cr_eff:.2f} drifts from nominal {cr}")
        check(bn.supports_wcc is False, f"CR{cr}: topk_sparse must not support WCC")
        check(tuple(bn.output_shape_) == CUT_SHAPE, f"CR{cr}: output shape {bn.output_shape_} != {CUT_SHAPE}")


def test_sparsity_and_correctness():
    """Keeps exactly k_s nonzeros/row at the top-k_s |h| positions; values intact; dense shape."""
    cr = 8
    ks = EXPECTED[cr][0]
    bn = bottlenecks.make_bottleneck("topk_sparse", CUT_SHAPE, compression_ratio=cr)
    h = tf.random.normal((6,) + CUT_SHAPE)
    z = bn(h, training=True)
    check(tuple(z.shape[1:]) == CUT_SHAPE, f"payload must stay dense/spatial, got {tuple(z.shape)}")

    z_flat = tf.reshape(z, [6, D]).numpy()
    h_flat = tf.reshape(h, [6, D]).numpy()
    nnz = (z_flat != 0).sum(axis=1)
    check(bool((nnz == ks).all()), f"each row must keep exactly {ks} nonzeros, got {nnz.tolist()}")

    # kept positions must be exactly the top-k_s by magnitude, with original values preserved
    idx_exp = np.argsort(-np.abs(h_flat), axis=1)[:, :ks]
    mask_exp = np.zeros_like(h_flat)
    np.put_along_axis(mask_exp, idx_exp, 1.0, axis=1)
    check(np.allclose(z_flat, h_flat * mask_exp),
          "kept coords must equal the exact top-k_s magnitude entries of the input")


def test_fsha_constructs_and_steps():
    """FSHA builds with bottleneck_mode=topk_sparse and runs a train step (dense -> raw decoder/D)."""
    rng = np.random.default_rng(123)
    x = rng.uniform(-1.0, 1.0, size=(8, 32, 32, 3)).astype("float32")
    y = rng.integers(0, 10, size=(8,), dtype=np.int32)
    xpriv = tf.data.Dataset.from_tensor_slices((x, y))
    xpub = tf.data.Dataset.from_tensor_slices((x, y))
    hparams = {
        "lr_f": 1e-4, "lr_tilde": 1e-4, "lr_D": 1e-4, "WGAN": False,
        "bottleneck_mode": "topk_sparse", "compression_ratio": 8.0,
        "bottleneck_k": None, "wcc_lambda": 0.0,
    }
    attack = FSHA.FSHA(xpriv, xpub, 3, 2, hparams)
    check(attack.bottleneck.k_s == EXPECTED[8][0], "FSHA bottleneck k_s mismatch")
    check(tuple(attack.bottleneck.output_shape_) == CUT_SHAPE,
          "FSHA defended shape must equal the raw cut (dense payload)")
    log = attack(2, log_frequency=1, verbose=False, progress_bar=False)
    check(np.isfinite(log).all(), f"training log must be finite, got {log}")
    sample = next(iter(xpriv.batch(2, drop_remainder=True)))[0]
    recon, control = attack.attack(sample)
    check(recon.shape == sample.shape, f"recon shape {recon.shape} != {sample.shape}")


def main():
    tests = [test_matched_byte_payload, test_sparsity_and_correctness, test_fsha_constructs_and_steps]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(tests)} tests FAILED")
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
