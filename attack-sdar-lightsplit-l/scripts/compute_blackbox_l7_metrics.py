import csv
import math
import os
import re

import lpips
import numpy as np
import torch
from PIL import Image


ROOT = os.path.dirname(__file__)
MODEL = "resnet"
DATASET = "cifar10"
LEVEL = 7
RUN = 1


def _log_dir(setting, method, cr=None):
    setting = setting.lower()
    if method == "raw":
        return "blackbox_l7_logs" if setting == "vsl" else "blackbox_usl_l7_cr8_logs"
    if method == "fixed_random":
        if cr == 8:
            return "blackbox_l7_logs" if setting == "vsl" else "blackbox_usl_l7_cr8_logs"
        return f"blackbox_{setting}_l7_cr{cr}_logs"
    return f"blackbox_{setting}_l7_{method}_cr{cr}_logs"


def _log_name(setting, method, cr=None, wcc=None):
    setting = setting.lower()
    if method == "raw":
        return f"{setting}_{MODEL}_{DATASET}_l{LEVEL}_run{RUN}_raw.log"
    return (
        f"{setting}_{MODEL}_{DATASET}_l{LEVEL}_run{RUN}_"
        f"{method}_cr{cr}_wcc{wcc:g}.log"
    )


def _recon_path(setting, method, cr=None, wcc=None):
    setting = setting.lower()
    if method == "raw":
        return f"sdar_results/{setting}/{DATASET}/fig/{MODEL}_l{LEVEL}_run{RUN}.png"

    if wcc and wcc != 0:
        name = f"cr{cr}_wcc{wcc:g}_{MODEL}_l{LEVEL}_run{RUN}.png"
    else:
        name = f"cr{cr}_{MODEL}_l{LEVEL}_run{RUN}.png"
    return f"defense_results/{method}/{setting}/{DATASET}/fig/{name}"


def build_runs():
    rows = []
    for setting in ["VSL", "USL"]:
        rows.append({
            "setting": setting,
            "method": "raw",
            "method_label": "Raw",
            "cr": None,
            "wcc": None,
            "log": os.path.join(_log_dir(setting, "raw"), _log_name(setting, "raw")),
            "recon": _recon_path(setting, "raw"),
        })

        for cr in [8, 16, 32]:
            for wcc in [0.0, 0.01, 0.1]:
                rows.append({
                    "setting": setting,
                    "method": "fixed_random",
                    "method_label": "\\textsc{LightSplit-F}",
                    "cr": cr,
                    "wcc": wcc,
                    "log": os.path.join(
                        _log_dir(setting, "fixed_random", cr),
                        _log_name(setting, "fixed_random", cr, wcc),
                    ),
                    "recon": _recon_path(setting, "fixed_random", cr, wcc),
                })

        for cr in [8, 16, 32]:
            rows.append({
                "setting": setting,
                "method": "learned_1x1",
                "method_label": "Learned $1\\times1$",
                "cr": cr,
                "wcc": 0.0,
                "log": os.path.join(
                    _log_dir(setting, "learned_1x1", cr),
                    _log_name(setting, "learned_1x1", cr, 0.0),
                ),
                "recon": _recon_path(setting, "learned_1x1", cr, 0.0),
            })

    return rows


def parse_log(path):
    text = open(path, "r", encoding="utf-8").read()
    if "Experiment done. Saving history" not in text:
        raise ValueError(f"Run is not complete: {path}")

    number = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    avg_mse = float(re.search(rf"Average MSE over all client's images: {number}", text).group(1))
    avg_ssim = float(re.search(rf"Average SSIM over all client's images: {number}", text).group(1))
    attack_match = re.search(
        rf"Attack MSE on batch: {number}, on these examples: {number}",
        text,
    )
    return {
        "mse": avg_mse,
        "ssim": avg_ssim,
        "psnr": -10.0 * math.log10(avg_mse),
        "batch_mse": float(attack_match.group(1)),
        "example_mse": float(attack_match.group(2)),
    }


def split_row_grid(path, n=20, tile=64):
    img = Image.open(path).convert("RGB")
    width, height = img.size
    tiles = []
    for i in range(n):
        x0 = round(i * width / n)
        x1 = round((i + 1) * width / n)
        tile_img = img.crop((x0, 0, x1, height)).resize((tile, tile), Image.Resampling.BILINEAR)
        tiles.append(np.asarray(tile_img).astype(np.float32) / 255.0)
    return np.stack(tiles, axis=0)


def lpips_for_grid(lpips_model, originals, recon):
    orig_t = torch.from_numpy(originals * 2.0 - 1.0).permute(0, 3, 1, 2)
    recon_t = torch.from_numpy(recon * 2.0 - 1.0).permute(0, 3, 1, 2)
    with torch.no_grad():
        return float(lpips_model(orig_t, recon_t).view(-1).mean().item())


def format_cr(value):
    return "--" if value is None else str(value)


def format_wcc(value):
    return "--" if value is None else f"{value:g}"


def format_label(value):
    return "--" if value is None else f"{value:.5f}"


def table_method(method_label):
    return method_label.replace("_", "\\_")


def write_csv(rows):
    csv_path = os.path.join(ROOT, "blackbox_l7_results_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "method",
                "compression_ratio",
                "wcc_lambda",
                "mse",
                "ssim",
                "psnr",
                "lpips_20",
                "batch_mse",
                "example_mse",
                "log",
                "recon",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "setting": row["setting"],
                "method": row["method"],
                "compression_ratio": "" if row["cr"] is None else row["cr"],
                "wcc_lambda": "" if row["wcc"] is None else f"{row['wcc']:g}",
                "mse": row["mse"],
                "ssim": row["ssim"],
                "psnr": row["psnr"],
                "lpips_20": row["lpips_20"],
                "batch_mse": row["batch_mse"],
                "example_mse": row["example_mse"],
                "log": row["log"],
                "recon": row["recon"],
            })
    return csv_path


def write_latex(rows):
    tex_path = os.path.join(ROOT, "blackbox_l7_results_table.tex")
    lines = [
        "% Generated from the black-box level-7 run logs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Level-7 SDAR black-box reconstruction results on CIFAR-10. \\textsc{LightSplit-F} rows use an $R$-known black-box attacker as a conservative stress test. Arrows point toward stronger privacy: higher MSE/LPIPS and lower SSIM/PSNR mean weaker reconstruction. PSNR is computed from the reported average MSE with image range $[0,1]$. LPIPS is computed on the saved 20-image visualization batch because the SDAR runs do not save full-dataset reconstruction tensors.}",
        "\\label{tab:sdar-blackbox-l7}",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Setting & Method & CR & $\\lambda_{\\mathrm{WCC}}$ & MSE $\\uparrow$ & SSIM $\\downarrow$ & PSNR (dB) $\\downarrow$ & LPIPS$_{20}$ $\\uparrow$ \\\\",
        "\\midrule",
    ]

    last_setting = None
    last_method = None
    for row in rows:
        if last_setting is not None and row["setting"] != last_setting:
            lines.append("\\midrule")
            last_method = None
        elif last_method is not None and row["method"] != last_method:
            lines.append("\\addlinespace[2pt]")

        lines.append(
            f"{row['setting']} & "
            f"{table_method(row['method_label'])} & "
            f"{format_cr(row['cr'])} & "
            f"{format_wcc(row['wcc'])} & "
            f"{row['mse']:.5f} & "
            f"{row['ssim']:.5f} & "
            f"{row['psnr']:.2f} & "
            f"{row['lpips_20']:.4f} \\\\"
        )
        last_setting = row["setting"]
        last_method = row["method"]

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return tex_path


def main():
    originals_path = os.path.join(ROOT, "sdar_results/vsl/cifar10/fig/resnet_l7_run1_originals.png")
    if not os.path.exists(originals_path):
        raise FileNotFoundError(
            "Missing originals grid. Run make_original_recon_comparison.py for the VSL raw run first."
        )

    originals = split_row_grid(originals_path)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    rows = []
    for run in build_runs():
        log_path = os.path.join(ROOT, run["log"])
        recon_path = os.path.join(ROOT, run["recon"])
        if not os.path.exists(log_path):
            raise FileNotFoundError(log_path)
        if not os.path.exists(recon_path):
            raise FileNotFoundError(recon_path)
        metrics = parse_log(log_path)
        recon = split_row_grid(recon_path)
        rows.append({
            **run,
            **metrics,
            "lpips_20": lpips_for_grid(lpips_model, originals, recon),
        })

    csv_path = write_csv(rows)
    tex_path = write_latex(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    for row in rows:
        print(
            f"{row['setting']:3s} {row['method']:13s} "
            f"CR={format_cr(row['cr']):>2s} WCC={format_wcc(row['wcc']):>4s} "
            f"MSE={row['mse']:.5f} SSIM={row['ssim']:.5f} "
            f"PSNR={row['psnr']:.2f} LPIPS20={row['lpips_20']:.4f}"
        )


if __name__ == "__main__":
    main()
