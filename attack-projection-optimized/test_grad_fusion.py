"""Unit tests for the adaptive activations+gradients attacker (Exp 6).

Standalone, CPU-only, deterministic, no network (random tensors only).
Run under the LightSplit conda env, from this folder:

    conda activate LightSplit && cd privileged_attacks && python test_grad_fusion.py

Exits 0 if all tests pass, 1 otherwise, so it can gate a commit:

    python test_grad_fusion.py && git commit ...

It guards the two promises of the change:
  (a) the new attack captures and fuses the returned cut-layer gradient g = dL/dz, and
  (b) copying the suite did not break the existing import/experiment surface.
"""
import sys
import traceback

import torch
from torch.utils.data import DataLoader, TensorDataset

from privileged_attack import (
    collect_activations_grads, GradFusionImageDecoder, run_exp4_nullspace,
    ActivationDecoder, OracleActivationAttacker,
)
from train_split import SplitModel
from client_models import MODEL_CONFIGS

torch.manual_seed(0)

DEVICE = torch.device("cpu")
D = MODEL_CONFIGS["cifar10"]["activation_dim"]   # 4096
K = 512                                          # CR 8x


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _orthonormal_R(d, k):
    """(d, k) matrix with orthonormal columns, same construction as the defense."""
    q, _ = torch.linalg.qr(torch.randn(d, k))
    return q[:, :k]


def test_grad_collection():
    """collect_activations_grads returns per-sample g = dL/dz: right shape, finite, nonzero."""
    n = 16
    imgs = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=8)

    model = SplitModel(dataset="cifar10", mode="fixed_random", k=K)
    data = collect_activations_grads(model, loader, DEVICE, "cifar10", max_samples=n)

    for key in ("z", "g", "x01", "y"):
        check(key in data, f"collect_activations_grads missing key '{key}'")
    check(tuple(data["g"].shape) == (n, K), f"g shape {tuple(data['g'].shape)} != {(n, K)}")
    check(tuple(data["z"].shape) == (n, K), f"z shape {tuple(data['z'].shape)} != {(n, K)}")
    check(tuple(data["x01"].shape) == (n, 3, 32, 32),
          f"x01 shape {tuple(data['x01'].shape)} unexpected")
    check(bool(torch.isfinite(data["g"]).all()), "g contains non-finite values")
    check(data["g"].abs().sum().item() > 0, "g is all zeros (gradient not captured)")


def test_fusion_decoder_shapes():
    """GradFusionImageDecoder maps (z, g) [+ pinv] -> [0,1] images of the right shape."""
    b = 4
    z = torch.randn(b, K)
    g = torch.randn(b, K)

    # activations + gradients only (no R)
    dec = GradFusionImageDecoder(K, D, R=None, fuse_pinv=False).eval()
    with torch.no_grad():
        out = dec(z, g)
    check(tuple(out.shape) == (b, 3, 32, 32), f"z+g out shape {tuple(out.shape)}")
    check(bool(torch.isfinite(out).all()), "z+g output non-finite")
    check(float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "z+g output not in [0,1]")

    # + known-R pseudoinverse channel
    R = _orthonormal_R(D, K)
    dec_p = GradFusionImageDecoder(K, D, R=R, fuse_pinv=True).eval()
    check(dec_p.fuse_pinv, "fuse_pinv flag did not take effect with R provided")
    with torch.no_grad():
        out_p = dec_p(z, g)
    check(tuple(out_p.shape) == (b, 3, 32, 32), f"z+g+pinv out shape {tuple(out_p.shape)}")
    check(bool(torch.isfinite(out_p).all()), "z+g+pinv output non-finite")


def test_gradient_channel_matters():
    """D(z, g) must differ from D(z, 0): the gradient channel is actually used."""
    b = 4
    z = torch.randn(b, K)
    g = torch.randn(b, K)
    dec = GradFusionImageDecoder(K, D, R=None, fuse_pinv=False).eval()
    with torch.no_grad():
        out_g = dec(z, g)
        out_0 = dec(z, torch.zeros_like(g))
    check(not torch.allclose(out_g, out_0, atol=1e-6),
          "decoder output ignores g (gradient channel has no effect)")


def test_no_regression_imports_and_exp4():
    """Copied suite still imports and Exp 4 (pure computation) still runs."""
    # classes import cleanly (symbol surface intact)
    _ = ActivationDecoder(D)
    _ = OracleActivationAttacker(_orthonormal_R(D, K), D, K)

    # Exp 4 null-space energy on synthetic isotropic h: frac_lost ~ (d-k)/d
    R = _orthonormal_R(D, K)
    data = {"h": torch.randn(64, D)}
    r4 = run_exp4_nullspace(R, data, K, D)
    check(r4.get("experiment") == "exp4_nullspace", "exp4 result mislabeled")
    frac = r4["frac_lost_mean"]
    check(0.0 < frac < 1.0, f"frac_lost_mean {frac} out of (0,1)")
    expected = (D - K) / D
    check(abs(frac - expected) < 0.1,
          f"frac_lost_mean {frac:.4f} far from theoretical {expected:.4f}")


TESTS = [
    ("grad_collection", test_grad_collection),
    ("fusion_decoder_shapes", test_fusion_decoder_shapes),
    ("gradient_channel_matters", test_gradient_channel_matters),
    ("no_regression_imports_and_exp4", test_no_regression_imports_and_exp4),
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
