#!/usr/bin/env python3
"""Collect SDAR black-box bottleneck metrics for one split level.

This is the level-aware version of compute_blackbox_l7_metrics.py. It expects
the same run grid used in the paper:

  raw;
  fixed_random CR={8,16,32}, WCC={0,0.01,0.1};
  learned_1x1 CR={8,16,32}, WCC=0;

The script intentionally fails on missing runs or missing full-dataset metrics
so partial tables do not accidentally enter the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
MODEL_DEFAULT = "resnet"
DATASET_DEFAULT = "cifar10"
RUN_DEFAULT = 1
CRS = [8, 16, 32]
WCCS = [0.0, 0.01, 0.1]
SETTINGS = ["VSL", "USL"]
LEVEL_GEOMETRY = {
    4: ("16x16x32", 8192, "4/2/1"),
    6: ("16x16x32", 8192, "4/2/1"),
    7: ("8x8x64", 4096, "8/4/2"),
}


@dataclass(frozen=True)
class RunSpec:
    setting: str
    method: str
    method_label: str
    cr: Optional[int]
    wcc: Optional[float]
    log_candidates: Sequence[Path]
    recon: Path


def _float_tag(value: float) -> str:
    return f"{value:g}"


def _log_name(setting: str, model: str, dataset: str, level: int, run: int, method: str, cr: Optional[int], wcc: Optional[float]) -> str:
    setting = setting.lower()
    if method == "raw":
        return f"{setting}_{model}_{dataset}_l{level}_run{run}_raw.log"
    return f"{setting}_{model}_{dataset}_l{level}_run{run}_{method}_cr{cr}_wcc{_float_tag(wcc or 0.0)}.log"


def _raw_log_candidates(setting: str, model: str, dataset: str, level: int, raw_run: int) -> List[Path]:
    setting_l = setting.lower()
    log_name = _log_name(setting_l, model, dataset, level, raw_run, "raw", None, None)
    candidates = [
        SCRIPT_DIR / f"blackbox_{setting_l}_l{level}_raw_rerun_logs" / log_name,
        SCRIPT_DIR / f"blackbox_l{level}_raw_rerun_logs" / log_name,
        SCRIPT_DIR / f"blackbox_{setting_l}_l{level}_raw_logs" / log_name,
        SCRIPT_DIR / f"blackbox_{setting_l}_l{level}_cr8_logs" / log_name,
        SCRIPT_DIR / f"blackbox_l{level}_logs" / log_name,
    ]
    if level == 7:
        candidates.extend([
            SCRIPT_DIR / ("blackbox_l7_logs" if setting_l == "vsl" else "blackbox_usl_l7_cr8_logs") / log_name,
        ])
    candidates.append(SCRIPT_DIR / "sdar_results" / setting_l / dataset / f"{model}_l{level}_run{raw_run}.txt")
    return candidates


def _defense_log_candidates(setting: str, model: str, dataset: str, level: int, run: int, method: str, cr: int, wcc: float) -> List[Path]:
    setting_l = setting.lower()
    log_name = _log_name(setting_l, model, dataset, level, run, method, cr, wcc)
    if method == "fixed_random":
        candidates = [
            SCRIPT_DIR / f"blackbox_{setting_l}_l{level}_cr{cr}_logs" / log_name,
        ]
        if level == 7 and cr == 8:
            candidates.append(SCRIPT_DIR / ("blackbox_l7_logs" if setting_l == "vsl" else "blackbox_usl_l7_cr8_logs") / log_name)
        return candidates
    return [
        SCRIPT_DIR / f"blackbox_{setting_l}_l{level}_{method}_cr{cr}_logs" / log_name,
    ]


def _recon_path(setting: str, dataset: str, model: str, level: int, run: int, method: str, cr: Optional[int], wcc: Optional[float]) -> Path:
    setting_l = setting.lower()
    if method == "raw":
        return SCRIPT_DIR / "sdar_results" / setting_l / dataset / "fig" / f"{model}_l{level}_run{run}.png"

    fig_dir = SCRIPT_DIR / "defense_results" / method / setting_l / dataset / "fig"
    if method == "fixed_random" and wcc not in (None, 0.0):
        name = f"cr{cr}_wcc{_float_tag(wcc)}_{model}_l{level}_run{run}.png"
    else:
        name = f"cr{cr}_{model}_l{level}_run{run}.png"
    return fig_dir / name


def build_runs(model: str, dataset: str, level: int, run: int, raw_run: int) -> List[RunSpec]:
    rows: List[RunSpec] = []
    for setting in SETTINGS:
        rows.append(RunSpec(
            setting=setting,
            method="raw",
            method_label="Raw",
            cr=None,
            wcc=None,
            log_candidates=_raw_log_candidates(setting, model, dataset, level, raw_run),
            recon=_recon_path(setting, dataset, model, level, raw_run, "raw", None, None),
        ))

        for cr in CRS:
            for wcc in WCCS:
                rows.append(RunSpec(
                    setting=setting,
                    method="fixed_random",
                    method_label="\\textsc{LightSplit-F}",
                    cr=cr,
                    wcc=wcc,
                    log_candidates=_defense_log_candidates(setting, model, dataset, level, run, "fixed_random", cr, wcc),
                    recon=_recon_path(setting, dataset, model, level, run, "fixed_random", cr, wcc),
                ))

        for cr in CRS:
            rows.append(RunSpec(
                setting=setting,
                method="learned_1x1",
                method_label="Learned $1\\times1$",
                cr=cr,
                wcc=0.0,
                log_candidates=_defense_log_candidates(setting, model, dataset, level, run, "learned_1x1", cr, 0.0),
                recon=_recon_path(setting, dataset, model, level, run, "learned_1x1", cr, 0.0),
            ))

    return rows


def find_existing(candidates: Sequence[Path], description: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    choices = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {description}. Tried:\n  {choices}")


def parse_log(path: Path) -> Dict[str, Optional[float]]:
    text = path.read_text(encoding="utf-8")
    if "Experiment done. Saving history" not in text:
        raise ValueError(f"Run is not complete: {path}")

    number = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"

    def required(pattern: str, label: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Missing {label} in {path}")
        return float(match.group(1))

    avg_mse = required(rf"Average MSE over all client's images: {number}", "average MSE")
    avg_ssim = required(rf"Average SSIM over all client's images: {number}", "average SSIM")
    attack_match = re.search(rf"Attack MSE on batch: {number}, on these examples: {number}", text)

    return {
        "mse": avg_mse,
        "ssim": avg_ssim,
        "psnr": -10.0 * math.log10(avg_mse),
        "batch_mse": float(attack_match.group(1)) if attack_match else None,
        "example_mse": float(attack_match.group(2)) if attack_match else None,
    }


def split_row_grid(path: Path, n: int = 20, tile: int = 64) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    tiles = []
    for i in range(n):
        x0 = round(i * width / n)
        x1 = round((i + 1) * width / n)
        tile_img = img.crop((x0, 0, x1, height)).resize((tile, tile), Image.Resampling.BILINEAR)
        tiles.append(np.asarray(tile_img).astype(np.float32) / 255.0)
    return np.stack(tiles, axis=0)


def lpips_for_grid(lpips_model: object, originals: np.ndarray, recon: np.ndarray) -> float:
    import torch

    orig_t = torch.from_numpy(originals * 2.0 - 1.0).permute(0, 3, 1, 2)
    recon_t = torch.from_numpy(recon * 2.0 - 1.0).permute(0, 3, 1, 2)
    with torch.no_grad():
        return float(lpips_model(orig_t, recon_t).view(-1).mean().item())


def ensure_originals(args: argparse.Namespace) -> Path:
    originals_path = SCRIPT_DIR / "sdar_results" / "vsl" / args.dataset / "fig" / f"{args.model}_l{args.level}_run{args.raw_run}_originals.png"
    if originals_path.is_file():
        return originals_path
    if not args.make_originals:
        raise FileNotFoundError(
            f"Missing originals grid: {originals_path}. "
            "Run make_original_recon_comparison.py or pass --make-originals."
        )

    raw_recon = _recon_path("vsl", args.dataset, args.model, args.level, args.raw_run, "raw", None, None)
    if not raw_recon.is_file():
        raise FileNotFoundError(f"Cannot create originals grid because raw VSL figure is missing: {raw_recon}")

    from make_original_recon_comparison import make_comparison

    comparison_path = raw_recon.with_name(f"{args.model}_l{args.level}_run{args.raw_run}_original_vs_recon.png")
    make_comparison(
        dataset=args.dataset,
        recon_fig=str(raw_recon),
        output=str(comparison_path),
        originals_output=str(originals_path),
        batch_size=args.batch_size,
        start=10,
        count=20,
    )
    return originals_path


def format_cr(value: Optional[int]) -> str:
    return "--" if value is None else str(value)


def format_wcc(value: Optional[float]) -> str:
    return "--" if value is None else _float_tag(value)


def format_optional(value: Optional[float], digits: int = 5) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def table_method(method_label: str) -> str:
    return method_label.replace("_", "\\_")


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
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
                "wcc_lambda": "" if row["wcc"] is None else _float_tag(float(row["wcc"])),
                "mse": row["mse"],
                "ssim": row["ssim"],
                "psnr": row["psnr"],
                "lpips_20": row["lpips_20"],
                "batch_mse": "" if row["batch_mse"] is None else row["batch_mse"],
                "example_mse": "" if row["example_mse"] is None else row["example_mse"],
                "log": row["log"],
                "recon": row["recon"],
            })


def setup_comments(level: int, model: str, dataset: str, run: int, raw_run: int) -> List[str]:
    shape, flat_dim, learned_channels = LEVEL_GEOMETRY.get(level, ("unknown", -1, "level-dependent"))
    if flat_dim > 0:
        k_values = ", ".join(f"CR {cr}: k={flat_dim // cr}" for cr in CRS)
    else:
        k_values = "computed by the SDAR bottleneck helper"
    return [
        f"% Generated by SDAR_SplitNN/scripts/compute_blackbox_level_metrics.py on {date.today().isoformat()}.",
        f"% Experiment setup: SDAR black-box reconstruction on {dataset} with {model}, split level {level}. Defended rows use run {run}; raw rows use run {raw_run}.",
        "% Both VSL and USL use the upstream SDAR defaults: 20,000 iterations for CIFAR levels <=7, batch size 128, Adam, and the released simulator/decoder/discriminator hyperparameters.",
        "% Raw rows are the original SDAR path. Defended rows insert the bottleneck after f(x) and apply the same transform to simulator features e(x').",
        "% Compression ratio rho is defined against the flat smashed activation dimension d, so rho=d/k is comparable across split levels.",
        f"% Level {level} geometry: smashed tensor {shape}, flat dimension d={flat_dim}. Fixed-random/top-k nominal payloads: {k_values}.",
        f"% Learned 1x1 uses channel bottlenecks {learned_channels} for CR 8/16/32 at this level.",
        "% \\textsc{SVD-rank-$k$} is batch-centered and has effective rank capped at B-1=127 for batch size 128.",
        "% LightSplit-F rows use an R-known black-box attacker as a conservative stress test; WCC is swept only for LightSplit-F.",
        "% LPIPS_20 is computed on the saved 20-image visualization batch because SDAR does not save full reconstruction tensors.",
    ]


def write_latex(rows: Sequence[Dict[str, object]], path: Path, level: int, model: str, dataset: str, run: int, raw_run: int) -> None:
    shape, flat_dim, _ = LEVEL_GEOMETRY.get(level, ("", -1, ""))
    geometry = f" The level-{level} smashed tensor is ${shape}$ with flat dimension $d={flat_dim}$." if flat_dim > 0 else ""
    lines = setup_comments(level, model, dataset, run, raw_run) + [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{Level-{level} SDAR black-box reconstruction results on CIFAR-10.{geometry} \\textsc{{LightSplit-F}} rows use an $R$-known black-box attacker as a conservative stress test. Arrows point toward stronger privacy: higher MSE/LPIPS and lower SSIM/PSNR mean weaker reconstruction. PSNR is computed from the reported average MSE with image range $[0,1]$. LPIPS is computed on the saved 20-image visualization batch.}}",
        f"\\label{{tab:sdar-blackbox-l{level}}}",
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
            f"{table_method(str(row['method_label']))} & "
            f"{format_cr(row['cr'])} & "
            f"{format_wcc(row['wcc'])} & "
            f"{float(row['mse']):.5f} & "
            f"{float(row['ssim']):.5f} & "
            f"{float(row['psnr']):.2f} & "
            f"{float(row['lpips_20']):.4f} \\\\"
        )
        last_setting = row["setting"]
        last_method = row["method"]

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(args: argparse.Namespace) -> List[Dict[str, object]]:
    import lpips

    originals_path = ensure_originals(args)
    originals = split_row_grid(originals_path)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()

    rows: List[Dict[str, object]] = []
    for spec in build_runs(args.model, args.dataset, args.level, args.run, args.raw_run):
        log_path = find_existing(spec.log_candidates, f"log for {spec.setting} {spec.method} CR={spec.cr} WCC={spec.wcc}")
        if not spec.recon.is_file():
            raise FileNotFoundError(f"Missing reconstruction PNG: {spec.recon}")
        metrics = parse_log(log_path)
        recon = split_row_grid(spec.recon)
        rows.append({
            "setting": spec.setting,
            "method": spec.method,
            "method_label": spec.method_label,
            "cr": spec.cr,
            "wcc": spec.wcc,
            "log": str(log_path.relative_to(SCRIPT_DIR)),
            "recon": str(spec.recon.relative_to(SCRIPT_DIR)),
            **metrics,
            "lpips_20": lpips_for_grid(lpips_model, originals, recon),
        })

    expected = len(SETTINGS) * (1 + len(CRS) * len(WCCS) + len(CRS) + len(CRS))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, collected {len(rows)} rows.")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SDAR black-box metrics for a split level.")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--run", type=int, default=RUN_DEFAULT)
    parser.add_argument("--raw-run", type=int, default=None, help="Run id for raw rows; defaults to --run.")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--make-originals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--tex", type=Path, default=None)
    parser.add_argument("--paper-tex", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_run is None:
        args.raw_run = args.run
    csv_path = args.csv or SCRIPT_DIR / f"blackbox_l{args.level}_results_metrics.csv"
    tex_path = args.tex or SCRIPT_DIR / f"blackbox_l{args.level}_results_table.tex"
    paper_tex_path = args.paper_tex or REPO_ROOT / "paper" / "tables" / f"sdar_l{args.level}_results_table.tex"

    rows = collect(args)
    write_csv(rows, csv_path)
    write_latex(rows, tex_path, args.level, args.model, args.dataset, args.run, args.raw_run)
    write_latex(rows, paper_tex_path, args.level, args.model, args.dataset, args.run, args.raw_run)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {paper_tex_path}")
    for row in rows:
        print(
            f"{row['setting']:3s} {row['method']:13s} "
            f"CR={format_cr(row['cr']):>2s} WCC={format_wcc(row['wcc']):>4s} "
            f"MSE={float(row['mse']):.5f} SSIM={float(row['ssim']):.5f} "
            f"PSNR={float(row['psnr']):.2f} LPIPS20={float(row['lpips_20']):.4f}"
        )


if __name__ == "__main__":
    main()
