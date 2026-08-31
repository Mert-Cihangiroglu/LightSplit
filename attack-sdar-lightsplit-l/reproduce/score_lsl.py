#!/usr/bin/env python3
"""Score the LightSplit-L SDAR runs (USL level 7) into tab:sdar_lsl.

  python score_lsl.py --logs DIR [--logs DIR ...] --scripts <this folder>/scripts --out tab_sdar_lsl.csv

Needs torch + lpips. Per CR: SSIM and MSE from each completed run log
(PSNR = -10 log10 MSE), LPIPS_20 between the run's saved 20-image grid and the
originals grid; the row is the mean over runs.
"""
import argparse, csv, glob, math, os, re
from pathlib import Path
import numpy as np
from PIL import Image

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def strip_grid(path, n=20, tile=64):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return np.stack([np.asarray(img.crop((round(i * w / n), 0, round((i + 1) * w / n), h))
                                .resize((tile, tile), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
                     for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", action="append", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--originals", help="originals strip PNG (default: the level-7 originals under --scripts)")
    args = ap.parse_args()
    import torch, lpips
    model = lpips.LPIPS(net="alex", verbose=False).eval()
    orig_png = Path(args.originals) if args.originals else \
        Path(args.scripts) / "sdar_results/vsl/cifar10/fig/resnet_l7_run1_originals.png"
    og = torch.from_numpy(strip_grid(orig_png) * 2 - 1).permute(0, 3, 1, 2)

    rows = []
    for cr in (8, 16, 32):
        per = []
        for d in args.logs:
            for log in sorted(glob.glob(os.path.join(d, "**", f"*frll_cr{cr}_run*.log"), recursive=True)) + \
                       sorted(glob.glob(os.path.join(d, "**", f"*cr{cr}_run*.log"), recursive=True)):
                t = open(log, errors="ignore").read()
                if "fixed_random_ll" not in t or "Experiment done" not in t:
                    continue
                mse = float(re.findall(rf"Average MSE over all client's images: {NUM}", t)[-1])
                ssim = float(re.findall(rf"Average SSIM over all client's images: {NUM}", t)[-1])
                m = re.search(r"History saved to (\S+)", t)
                png = None
                if m:
                    p = Path(m.group(1))
                    if not p.is_absolute():
                        p = Path(args.scripts) / p
                    png = p.parent / "fig" / p.name.replace(".npy", ".png")
                if png is None or not png.exists():
                    continue
                if any(x["log"] == log for x in per):
                    continue
                rg = torch.from_numpy(strip_grid(png) * 2 - 1).permute(0, 3, 1, 2)
                with torch.no_grad():
                    lp = float(model(og, rg).view(-1).mean())
                per.append(dict(log=log, mse=mse, ssim=ssim, psnr=-10 * math.log10(mse), lpips=lp))
        if per:
            rows.append([cr, len(per),
                         f"{sum(p['ssim'] for p in per)/len(per):.5f}",
                         f"{sum(p['psnr'] for p in per)/len(per):.4f}",
                         f"{sum(p['lpips'] for p in per)/len(per):.6f}"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cr", "n_runs", "ssim", "psnr", "lpips"])
        w.writerows(rows)
    print(f"[score_lsl] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
