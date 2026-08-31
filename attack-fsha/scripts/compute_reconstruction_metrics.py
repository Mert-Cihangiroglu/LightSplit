#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_DIR / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "reconstruction_quality_metrics.csv"


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_safe)
        handle.write("\n")


def read_json(path):
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def to_unit(images):
    return np.clip((images.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)


def mse_per_image(x, y):
    axes = tuple(range(1, x.ndim))
    return np.mean((x - y) ** 2, axis=axes)


def psnr_per_image(mse):
    return np.asarray([
        float("inf") if value <= 0 else 10.0 * math.log10(1.0 / float(value))
        for value in mse
    ], dtype=np.float64)


def gaussian_kernel(window_size=11, sigma=1.5):
    import torch

    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def ssim_per_image(x, y, batch_size=32):
    import torch
    import torch.nn.functional as F

    x_t = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float()
    y_t = torch.from_numpy(np.transpose(y, (0, 3, 1, 2))).float()
    channels = x_t.shape[1]
    kernel = gaussian_kernel().to(x_t.device)
    weight = kernel.expand(channels, 1, -1, -1).contiguous()
    padding = kernel.shape[-1] // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []

    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            xb = x_t[start : start + batch_size]
            yb = y_t[start : start + batch_size]
            mu_x = F.conv2d(xb, weight, padding=padding, groups=channels)
            mu_y = F.conv2d(yb, weight, padding=padding, groups=channels)
            mu_x2 = mu_x * mu_x
            mu_y2 = mu_y * mu_y
            mu_xy = mu_x * mu_y

            sigma_x2 = F.conv2d(xb * xb, weight, padding=padding, groups=channels) - mu_x2
            sigma_y2 = F.conv2d(yb * yb, weight, padding=padding, groups=channels) - mu_y2
            sigma_xy = F.conv2d(xb * yb, weight, padding=padding, groups=channels) - mu_xy

            ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
                (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
            )
            scores.append(ssim_map.mean(dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(scores, axis=0)


def lpips_per_image(x_minus_one_one, y_minus_one_one, batch_size=16):
    import torch
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    x_t = torch.from_numpy(np.transpose(x_minus_one_one.astype(np.float32), (0, 3, 1, 2)))
    y_t = torch.from_numpy(np.transpose(y_minus_one_one.astype(np.float32), (0, 3, 1, 2)))
    scores = []
    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            xb = x_t[start : start + batch_size].to(device)
            yb = y_t[start : start + batch_size].to(device)
            value = loss_fn(xb, yb)
            scores.append(value.reshape(-1).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}


def nearest_manifest(run_dir):
    for parent in [run_dir, *run_dir.parents]:
        manifest = parent / "manifest.json"
        if manifest.exists():
            return manifest
        if parent == DEFAULT_RESULTS_DIR.parent:
            break
    return None


def metrics_for_run(run_dir, batch_size, compute_lpips=True):
    recon_path = run_dir / "reconstructions.npz"
    payload = np.load(recon_path)
    original = payload["original"]
    recovered = payload["recovered"]

    original_unit = to_unit(original)
    recovered_unit = to_unit(recovered)
    original_lpips = original_unit * 2.0 - 1.0
    recovered_lpips = recovered_unit * 2.0 - 1.0

    mse = mse_per_image(original_unit, recovered_unit)
    psnr = psnr_per_image(mse)
    ssim = ssim_per_image(original_unit, recovered_unit, batch_size=batch_size)

    lpips_values = None
    lpips_error = None
    if compute_lpips:
        try:
            lpips_values = lpips_per_image(original_lpips, recovered_lpips, batch_size=batch_size)
        except Exception as exc:
            lpips_error = repr(exc)

    config = read_json(run_dir / "config.json")
    existing_metrics = read_json(run_dir / "metrics.json")
    manifest_path = nearest_manifest(run_dir)
    manifest = read_json(manifest_path) if manifest_path else {}

    metrics = {
        "case_id": config.get("case_id") or existing_metrics.get("case_id") or run_dir.name,
        "dataset": config.get("dataset") or manifest.get("args", {}).get("dataset"),
        "run_dir": str(run_dir),
        "num_images": int(original.shape[0]),
        "image_range": "[0, 1]",
        "reference": "original",
        "prediction": "recovered",
        "mse": summarize(mse),
        "ssim": summarize(ssim),
        "psnr": summarize(psnr),
        "lpips_alex": summarize(lpips_values) if lpips_values is not None else {"mean": None, "std": None},
        "lpips_error": lpips_error,
        "metric_note": "MSE, SSIM, and PSNR are computed after mapping saved [-1, 1] images to [0, 1]. LPIPS uses AlexNet/v0.1 on recovered images when lpips is installed.",
    }
    dump_json(run_dir / "reconstruction_quality_metrics.json", metrics)
    return metrics


def write_csv(path, rows):
    fields = [
        "case_id",
        "dataset",
        "run_dir",
        "num_images",
        "mse_mean",
        "mse_std",
        "ssim_mean",
        "ssim_std",
        "psnr_mean",
        "psnr_std",
        "lpips_alex_mean",
        "lpips_alex_std",
        "lpips_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "dataset": row.get("dataset"),
                    "run_dir": row.get("run_dir"),
                    "num_images": row.get("num_images"),
                    "mse_mean": row["mse"]["mean"],
                    "mse_std": row["mse"]["std"],
                    "ssim_mean": row["ssim"]["mean"],
                    "ssim_std": row["ssim"]["std"],
                    "psnr_mean": row["psnr"]["mean"],
                    "psnr_std": row["psnr"]["std"],
                    "lpips_alex_mean": row["lpips_alex"]["mean"],
                    "lpips_alex_std": row["lpips_alex"]["std"],
                    "lpips_error": row.get("lpips_error"),
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute original-vs-recovered reconstruction metrics.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    recon_paths = sorted(Path(args.results_dir).glob("bottleneck_grid/*/**/reconstructions.npz"))
    rows = []
    for recon_path in recon_paths:
        run_dir = recon_path.parent
        print("[METRICS]", run_dir)
        rows.append(metrics_for_run(run_dir, args.batch_size, compute_lpips=not args.no_lpips))
    write_csv(Path(args.output), rows)
    print(Path(args.output))


if __name__ == "__main__":
    main()
