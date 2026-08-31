#!/usr/bin/env python3
"""
Projection x WCC grid (utility + privacy) from the canonical SDAR_SplitNN harness.

For each (compression_ratio CR) x (wcc_lambda) cell of LightSplit-F (bottleneck_mode
= fixed_random), this assembles ONE table with BOTH:
  * utility  : the foreground model's classification accuracy
               ("Average label accuracy over all client's images" in the run log)
  * privacy  : SDAR reconstruction MSE / SSIM / PSNR / LPIPS
               (from blackbox_l4_results_metrics.csv)

Decomposes the privacy gain into projection alone (lambda=0) vs
projection+WCC (lambda>0).

Two modes:
  (default)  ASSEMBLE  : parse the existing logs + metrics CSV, write the grid CSV.
  --run      GENERATE  : for any cell whose log is missing, invoke run_sdar.py
                         (TensorFlow; needs a GPU and the SDAR env) then assemble.

Usage:
  python make_wcc_cr_grid.py                          # assemble USL grid from disk
  python make_wcc_cr_grid.py --settings usl vsl       # both settings
  python make_wcc_cr_grid.py --run --gpu 1            # (re)generate missing cells
  python make_wcc_cr_grid.py --cr-list 8 16 32 --wcc-list 0 0.01 0.1
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_CSV = SCRIPT_DIR / "blackbox_l4_results_metrics.csv"

_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
ACC_RE = re.compile(r"Average label accuracy over all client's images:\s*" + _FLOAT)
SSIM_RE = re.compile(r"Average SSIM over all client's images:\s*" + _FLOAT)
MSE_RE = re.compile(r"Average MSE over all client's images:\s*" + _FLOAT)


def fmt_wcc(w):
    """Match run_sdar.py's f'{wcc:g}' tag (0 -> '0', 0.01 -> '0.01')."""
    return f"{float(w):g}"


def log_relpath(setting, cr, wcc):
    """Reconstruct the log path convention used by the blackbox sweeps."""
    tag = f"cr{int(cr)}_wcc{fmt_wcc(wcc)}"
    return (
        f"blackbox_{setting}_l4_cr{int(cr)}_logs/"
        f"{setting}_resnet_cifar10_l4_run1_fixed_random_{tag}.log"
    )


def parse_log_accuracy(path):
    """Return final label accuracy from a run log, or None if absent/missing."""
    p = SCRIPT_DIR / path
    if not p.exists():
        return None
    text = p.read_text(errors="ignore")
    m = ACC_RE.findall(text)
    return float(m[-1]) if m else None


def load_metrics_csv():
    """Index reconstruction metrics by (setting, method, cr, wcc)."""
    rows = {}
    if not METRICS_CSV.exists():
        return rows
    with open(METRICS_CSV) as f:
        for r in csv.DictReader(f):
            setting = (r.get("setting") or "").strip().lower()
            method = (r.get("method") or "").strip()
            cr = (r.get("compression_ratio") or "").strip()
            wcc = (r.get("wcc_lambda") or "").strip()
            key = (setting, method, cr, fmt_wcc(wcc) if wcc != "" else "")
            rows[key] = r
    return rows


def run_cell(setting, cr, wcc, gpu):
    """Invoke run_sdar.py for one cell, redirecting stdout to the log path."""
    out = SCRIPT_DIR / log_relpath(setting, cr, wcc)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "run_sdar.py", setting, "resnet", "cifar10", "4", "1",
        "--bottleneck_mode", "fixed_random",
        "--compression_ratio", str(int(cr)),
        "--wcc_lambda", fmt_wcc(wcc),
        "--print_to_stdout",
    ]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"  [run] {' '.join(cmd)}  -> {out.name}")
    with open(out, "w") as fh:
        subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)


def build(settings, cr_list, wcc_list, do_run, gpu):
    metrics = load_metrics_csv()
    grid = []
    for setting in settings:
        # raw (no-defense) reference row, if present in the CSV
        raw = metrics.get((setting, "raw", "", ""))
        raw_ssim = float(raw["ssim"]) if raw else None
        raw_acc = parse_log_accuracy(raw["log"]) if raw and raw.get("log") else None
        if raw:
            grid.append(dict(setting=setting, cr="raw", wcc="", acc=raw_acc,
                             mse=float(raw["mse"]), ssim=float(raw["ssim"]),
                             psnr=float(raw["psnr"]), lpips=float(raw["lpips_20"])))
        for cr in cr_list:
            for wcc in wcc_list:
                if do_run and not (SCRIPT_DIR / log_relpath(setting, cr, wcc)).exists():
                    run_cell(setting, cr, wcc, gpu)
                acc = parse_log_accuracy(log_relpath(setting, cr, wcc))
                m = metrics.get((setting, "fixed_random", str(int(cr)), fmt_wcc(wcc)))
                grid.append(dict(
                    setting=setting, cr=int(cr), wcc=fmt_wcc(wcc), acc=acc,
                    mse=float(m["mse"]) if m else None,
                    ssim=float(m["ssim"]) if m else None,
                    psnr=float(m["psnr"]) if m else None,
                    lpips=float(m["lpips_20"]) if m else None,
                ))
    # attach relative-to-raw metrics per setting
    raw_by_setting = {r["setting"]: r for r in grid if r["cr"] == "raw"}
    for r in grid:
        base = raw_by_setting.get(r["setting"])
        ra = base.get("acc") if base else None
        rs = base.get("ssim") if base else None
        r["dacc"] = (r["acc"] - ra) if (r["acc"] is not None and ra is not None) else None
        r["ssim_pct_raw"] = (100.0 * r["ssim"] / rs) if (r["ssim"] is not None and rs) else None
    return grid


def write_csv(grid, out_path):
    cols = ["setting", "cr", "wcc", "acc", "dacc", "mse", "ssim",
            "ssim_pct_raw", "psnr", "lpips"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in grid:
            w.writerow({c: row.get(c) for c in cols})


def print_table(grid):
    def s(x, nd=4):
        return "  n/a " if x is None else f"{x:.{nd}f}"
    def sd(x):  # signed delta in accuracy points
        return "  n/a " if x is None else f"{100*x:+6.1f}"
    def sp(x):  # ssim as % of raw
        return "  n/a " if x is None else f"{x:5.0f}%"
    cur = None
    for row in grid:
        if row["setting"] != cur:
            cur = row["setting"]
            print(f"\n=== {cur.upper()}  (LightSplit-F, ResNet, CIFAR-10, L4) ===")
            print(f"   utility (acc, vs-raw pp)  |  privacy (recon; lower=better, "
                  f"SSIM as % of raw)")
            print(f"{'CR':>4} {'lam':>5} | {'ACC':>7} {'dRaw':>6} | "
                  f"{'SSIM':>7} {'%raw':>5} {'PSNR':>6} {'LPIPS':>6} {'MSE':>6}")
            print("-" * 66)
        lam = row["wcc"] if row["wcc"] != "" else "-"
        print(f"{str(row['cr']):>4} {str(lam):>5} | {s(row['acc'],4):>7} "
              f"{sd(row['dacc']):>6} | {s(row['ssim']):>7} {sp(row['ssim_pct_raw']):>5} "
              f"{s(row['psnr'],1):>6} {s(row['lpips'],3):>6} {s(row['mse'],3):>6}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=["usl"],
                    choices=["usl", "vsl"])
    ap.add_argument("--cr-list", nargs="+", type=float, default=[8, 16, 32])
    ap.add_argument("--wcc-list", nargs="+", type=float, default=[0, 0.01, 0.1])
    ap.add_argument("--run", action="store_true",
                    help="generate missing cells via run_sdar.py (TF/GPU)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default=str(SCRIPT_DIR / "wcc_cr_grid.csv"))
    args = ap.parse_args()

    grid = build(args.settings, args.cr_list, args.wcc_list, args.run, args.gpu)
    write_csv(grid, args.out)
    print_table(grid)
    n_missing = sum(1 for r in grid if r["acc"] is None or r["ssim"] is None)
    print(f"\nWrote {len(grid)} rows to {args.out}"
          f"  ({n_missing} cell(s) missing data; use --run to fill).")


if __name__ == "__main__":
    main()
