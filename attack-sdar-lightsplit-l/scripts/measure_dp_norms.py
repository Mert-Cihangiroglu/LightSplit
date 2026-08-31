#!/usr/bin/env python3
"""Measure the median per-sample L2 norm of the SDAR bottleneck payload, to calibrate
the DP clip norm (clip = per-config median so the clip meaningfully bites and the
sigma*clip noise is scaled to the real signal).

Builds the SDAR client head ``f`` and the requested bottleneck at the given level,
pushes a few real CIFAR-10 batches through, and reports the median / mean / percentiles
of ||payload||_2 across samples. DP itself is OFF here (clip=0), so we see the raw
payload scale.

Usage:
  python measure_dp_norms.py cifar10 7 --mode none            # raw (identity) payload
  python measure_dp_norms.py cifar10 7 --mode fixed_random --compression_ratio 8
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tensorflow as tf
import data
from sdar.bottlenecks import make_bottleneck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
    ap.add_argument("level", type=int)
    ap.add_argument("--mode", default="none",
                    choices=["none", "fixed_random", "topk_sparse", "learned_1x1", "block_random"])
    ap.add_argument("--compression_ratio", type=float, default=1.0)
    ap.add_argument("--bottleneck_k", type=int, default=0)
    ap.add_argument("--block_size", type=int, default=2)
    ap.add_argument("--model", default="resnet", choices=["resnet", "plainnet"])
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    if args.model == "resnet":
        from models.resnet import make_f
    else:
        from models.plainnet import make_f

    # a handful of real client batches
    client_ds, _ = data.load_dataset(args.dataset, frac=1.0)
    ds = client_ds.batch(args.batch_size).take(args.n_batches)

    input_shape = client_ds.element_spec[0].shape
    f = make_f(args.level, input_shape)
    inter_shape = f.layers[-1].output_shape[1:]
    bottleneck = make_bottleneck(
        args.mode, inter_shape,
        compression_ratio=args.compression_ratio,
        bottleneck_k=args.bottleneck_k,
        block_size=args.block_size,
        dp_clip=0.0, dp_sigma=0.0,   # DP off: measure raw payload scale
    )

    norms = []
    for x, _ in ds:
        z_raw = f(x, training=True)          # match fg_train_step's training=True
        _, payload = bottleneck(z_raw, training=True)
        flat = tf.reshape(payload, [tf.shape(payload)[0], -1])
        n = tf.norm(flat, axis=1).numpy()
        norms.append(n)
    norms = np.concatenate(norms)

    print(f"mode={args.mode} level={args.level} cr={args.compression_ratio} "
          f"payload_dim={bottleneck.payload_dim}  (n={norms.size} samples)")
    print(f"  ||payload||_2  median={np.median(norms):.4f}  mean={norms.mean():.4f}  "
          f"std={norms.std():.4f}")
    print(f"  percentiles  p10={np.percentile(norms,10):.4f}  p50={np.percentile(norms,50):.4f}  "
          f"p90={np.percentile(norms,90):.4f}  max={norms.max():.4f}")
    print(f"  --> suggested --dp_clip_norm {np.median(norms):.3f}")


if __name__ == "__main__":
    main()
