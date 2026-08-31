#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)


INPUT_SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar100": (32, 32, 3),
    "tinyimagenet": (64, 64, 3),
    "stl10": (96, 96, 3),
}


def trainable_param_count(model):
    return int(sum(np.prod(variable.shape) for variable in model.trainable_variables))


def activation_dim(shape):
    return int(np.prod(shape))


def fixed_random_payload_dim(shape, compression_ratio, bottleneck_k):
    dim = activation_dim(shape)
    if bottleneck_k > 0:
        return max(1, min(dim, bottleneck_k))
    return max(1, min(dim, int(round(dim / compression_ratio))))


def make_client_model(model_type, level, input_shape, width):
    if model_type == "resnet":
        from models.resnet import make_f
    elif model_type == "plainnet":
        from models.plainnet import make_f
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return make_f(level, input_shape, width=width)


MODEL_WIDTHS = {
    "narrow": [8, 16, 32],
    "standard": [16, 32, 64],
    "wide": [32, 64, 128],
}


def conv_params(kernel, in_channels, out_channels, use_bias=False):
    params = kernel * kernel * in_channels * out_channels
    if use_bias:
        params += out_channels
    return params


def bn_params(channels):
    return 2 * channels


def resnet_basic_block_params(in_channels, planes, downsample=False):
    params = conv_params(3, in_channels, planes) + bn_params(planes)
    params += conv_params(3, planes, planes) + bn_params(planes)
    if downsample:
        params += conv_params(1, in_channels, planes) + bn_params(planes)
    return params


def plainnet_basic_block_params(in_channels, planes):
    params = conv_params(3, in_channels, planes) + bn_params(planes)
    params += conv_params(3, planes, planes) + bn_params(planes)
    return params


def downsampled_size(size):
    return (size + 1) // 2


def static_client_stats(model_type, level, input_shape, width):
    widths = MODEL_WIDTHS[width]
    height, width_px, channels = input_shape

    params = conv_params(3, channels, widths[0]) + bn_params(widths[0])
    channels = widths[0]

    if model_type == "resnet":
        for _ in range(3):
            params += resnet_basic_block_params(channels, widths[0])
        if level >= 4:
            params += resnet_basic_block_params(channels, widths[1], downsample=True)
            channels = widths[1]
            height = downsampled_size(height)
            width_px = downsampled_size(width_px)
        if level >= 5:
            params += resnet_basic_block_params(channels, widths[1])
        if level >= 6:
            params += resnet_basic_block_params(channels, widths[1])
        if level >= 7:
            params += resnet_basic_block_params(channels, widths[2], downsample=True)
            channels = widths[2]
            height = downsampled_size(height)
            width_px = downsampled_size(width_px)
        if level >= 8:
            params += resnet_basic_block_params(channels, widths[2])
        if level >= 9:
            params += resnet_basic_block_params(channels, widths[2])
    elif model_type == "plainnet":
        for _ in range(3):
            params += plainnet_basic_block_params(channels, widths[0])
        if level >= 4:
            params += plainnet_basic_block_params(channels, widths[1])
            channels = widths[1]
            height = downsampled_size(height)
            width_px = downsampled_size(width_px)
        if level >= 5:
            params += plainnet_basic_block_params(channels, widths[1])
        if level >= 6:
            params += plainnet_basic_block_params(channels, widths[1])
        if level >= 7:
            params += plainnet_basic_block_params(channels, widths[2])
            channels = widths[2]
            height = downsampled_size(height)
            width_px = downsampled_size(width_px)
        if level >= 8:
            params += plainnet_basic_block_params(channels, widths[2])
        if level >= 9:
            params += plainnet_basic_block_params(channels, widths[2])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return params, (height, width_px, channels)


def collect_rows(args):
    rows = []
    input_shape = INPUT_SHAPES[args.dataset]
    for level in args.levels:
        if args.backend == "static":
            params, smashed_shape = static_client_stats(args.model, level, input_shape, args.width)
        else:
            try:
                model = make_client_model(args.model, level, input_shape, args.width)
                params = trainable_param_count(model)
                smashed_shape = tuple(int(v) for v in model.output_shape[1:])
            except ModuleNotFoundError as exc:
                if args.backend != "auto" or exc.name != "tensorflow":
                    raise
                params, smashed_shape = static_client_stats(args.model, level, input_shape, args.width)
        payload_dim = fixed_random_payload_dim(
            smashed_shape,
            args.compression_ratio,
            args.bottleneck_k,
        )
        rows.append({
            "level": level,
            "client_params": params,
            "smashed_shape": "x".join(str(v) for v in smashed_shape),
            "activation_dim": activation_dim(smashed_shape),
            "fixed_random_payload_dim": payload_dim,
            "effective_compression_ratio": activation_dim(smashed_shape) / payload_dim,
        })
    return rows


def select_level(rows, target_params, mode):
    if target_params is None:
        return None
    if mode == "closest":
        return min(rows, key=lambda row: abs(row["client_params"] - target_params))
    if mode == "at_least":
        candidates = [row for row in rows if row["client_params"] >= target_params]
        return min(candidates, key=lambda row: row["client_params"]) if candidates else None
    if mode == "at_most":
        candidates = [row for row in rows if row["client_params"] <= target_params]
        return max(candidates, key=lambda row: row["client_params"]) if candidates else None
    raise ValueError(f"Unknown selection mode: {mode}")


def print_table(rows):
    headers = [
        "level",
        "client_params",
        "smashed_shape",
        "activation_dim",
        "fixed_random_payload_dim",
        "effective_compression_ratio",
    ]
    widths = {
        header: max(len(header), *(len(format_cell(row[header], header)) for row in rows))
        for header in headers
    }
    print(" ".join(header.rjust(widths[header]) for header in headers))
    print(" ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" ".join(format_cell(row[header], header).rjust(widths[header]) for header in headers))


def print_csv(rows):
    headers = [
        "level",
        "client_params",
        "smashed_shape",
        "activation_dim",
        "fixed_random_payload_dim",
        "effective_compression_ratio",
    ]
    print(",".join(headers))
    for row in rows:
        print(",".join(format_cell(row[header], header, csv=True) for header in headers))


def format_cell(value, header, csv=False):
    if header == "effective_compression_ratio":
        return f"{value:.4g}"
    if isinstance(value, int) and not csv:
        return f"{value:,}"
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect SDAR client-head parameter counts by split level."
    )
    parser.add_argument("--model", default="resnet", choices=["resnet", "plainnet"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
    parser.add_argument("--width", default="standard", choices=["standard", "wide", "narrow"])
    parser.add_argument("--levels", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--compression_ratio", type=float, default=8.0)
    parser.add_argument("--bottleneck_k", type=int, default=0)
    parser.add_argument("--target_params", type=int, default=None)
    parser.add_argument("--select", default="closest", choices=["closest", "at_least", "at_most"])
    parser.add_argument("--backend", default="auto", choices=["auto", "keras", "static"])
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compression_ratio <= 0.0:
        raise ValueError("compression_ratio must be positive.")
    if args.bottleneck_k < 0:
        raise ValueError("bottleneck_k must be non-negative.")

    rows = collect_rows(args)
    if args.csv:
        print_csv(rows)
    else:
        print_table(rows)

    selected = select_level(rows, args.target_params, args.select)
    if selected is not None:
        print()
        print(
            "selected_level="
            f"{selected['level']} "
            f"client_params={selected['client_params']} "
            f"selection={args.select} "
            f"target_params={args.target_params}"
        )
    elif args.target_params is not None:
        print()
        print(f"selected_level=none selection={args.select} target_params={args.target_params}")


if __name__ == "__main__":
    main()
