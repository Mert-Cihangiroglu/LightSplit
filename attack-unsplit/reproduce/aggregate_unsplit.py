#!/usr/bin/env python
"""Aggregate UnSplit runs into the paper's UnSplit tables.

Reads one or more run directories (each holding `results/summary.csv` plus the reconstruction
PNG pairs that `model_inversion_demo.py` writes), scores every reconstruction against its target
and writes

  <out>.csv           one row per (dataset, split, method, block, wcc, cr, seed, dp_sigma) cell:
                      acc, whole-image mse/ssim/psnr/lpips, foreground mse, and the ratios to the
                      Raw baseline of the same (dataset, split)
  <out>_headline.csv  the split-2 summary: accuracy range and worst (smallest) MSE_fg ratio over
                      CR (and lambda_WCC) per method

Metrics: whole-image MSE (the attack's own value from summary.csv, measured on float tensors before PNG
quantisation), and from the saved PNG pair: SSIM (7x7 uniform window, sample covariance, K1=0.01, K2=0.03,
data_range=1, i.e. the scikit-image defaults), PSNR, LPIPS-AlexNet on the image bilinearly
upscaled to 64x64; foreground MSE over pixels where the target exceeds 0.1.

Baselines: a single-image cell is divided by the single-image Raw cell (defense `none`) of the
same (dataset, split); a multi-image cell (the DP rows, 10 images) by the mean of the 10-image
Raw sigma=0 run (defense `raw`) of the same (dataset, split). The `*0` columns hold the metrics
of image index 0 alone (`mse_fg0_ratio` against the single-image Raw cell); the paper's DP rows
print those for the whole-image metrics and, at cut 4, for the MSE_fg ratio. If a cell was run
more than once the last row per image index wins.
"""
import argparse, csv, math, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image

ACT_DIM = {2: 1152, 4: 1024}
COLS = ["dataset", "split", "method", "block", "wcc", "cr", "k", "seed", "dp_sigma", "n_images", "acc",
        "mse", "mse_ratio", "ssim", "psnr", "lpips", "mse_fg", "mse_fg_ratio", "mse_fg_pct",
        "mse0", "ssim0", "psnr0", "lpips0", "mse_fg0_ratio"]


def load(p):
    return np.asarray(Image.open(p).convert("L")).astype(np.float32) / 255.0


def _box(x, w=7):
    p = w // 2
    xp = np.pad(x, p, mode="symmetric")
    c = np.pad(np.cumsum(np.cumsum(xp, 0), 1), ((1, 0), (1, 0)))
    H, W = x.shape
    return (c[w:w + H, w:w + W] - c[:H, w:w + W] - c[w:w + H, :W] + c[:H, :W]) / (w * w)


def ssim(x, y, w=7, dr=1.0):
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    cov = (w * w) / (w * w - 1)
    ux, uy = _box(x, w), _box(y, w)
    vx, vy, vxy = cov * (_box(x * x, w) - ux * ux), cov * (_box(y * y, w) - uy * uy), cov * (_box(x * y, w) - ux * uy)
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    p = (w - 1) // 2
    return float(S[p:-p, p:-p].mean())


_LPIPS = None


def lpips_dist(x, y):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
        for q in _LPIPS.parameters():
            q.requires_grad_(False)

    def t(a):
        tt = torch.from_numpy(np.stack([a] * 3, -1)).permute(2, 0, 1).unsqueeze(0)
        if min(tt.shape[-2:]) < 64:
            tt = torch.nn.functional.interpolate(tt, size=(64, 64), mode="bilinear", align_corners=False)
        return tt * 2 - 1

    with torch.no_grad():
        return float(_LPIPS(t(x), t(y)).item())


def metrics(recon, orig):
    m = {"mse": float(((recon - orig) ** 2).mean()), "ssim": ssim(orig, recon), "lpips": lpips_dist(orig, recon)}
    m["psnr"] = float(10 * np.log10(1.0 / m["mse"])) if m["mse"] > 0 else float("inf")
    mask = orig > 0.1
    m["mse_fg"] = float(((recon - orig) ** 2)[mask].mean()) if mask.any() else float("nan")
    return m


def png_pair(res, r):
    ds, d, sp, idx = r["dataset"], r["defense"], r["split"], r["idx"]
    if d == "none":
        tg = sorted(res.glob(f"{ds}_split{sp}_*_0_target.png"))
        return res / f"{ds}_split{sp}_{idx}.png", (tg[0] if tg else None)
    k = f"_k{r['k']}" if d != "raw" and r["k"] else ""
    b = f"_block{r['block_size']}" if d == "block_random" and r["block_size"] else ""
    w = f"_wcc{float(r['wcc_lambda'] or 0):g}" if d in ("fixed_random", "block_random") else ""
    dp = f"_dp{float(r['noise_multiplier']):g}" if r.get("clip_norm") not in ("", None) and float(r["clip_norm"]) > 0 else ""
    sd = f"_seed{r['seed']}" if r.get("seed") not in ("", None) else ""
    s = f"{ds}_split{sp}_{d}{k}{b}{w}{dp}{sd}_{idx}"
    return res / f"{s}.png", res / f"{s}_target.png"


def fmt(v, nd=4):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True, help="run directory containing summary.csv and the PNGs (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = defaultdict(dict)  # key -> {idx: (row, metrics)}
    for rd in args.results:
        res = Path(rd)
        for r in csv.DictReader(open(res / "summary.csv")):
            if r["dataset"] == "cifar":
                continue
            key = (r["dataset"], r["defense"], int(r["split"]), r.get("k", ""), f"{float(r.get('wcc_lambda') or 0):g}",
                   r.get("block_size", ""), r.get("clip_norm", ""), r.get("noise_multiplier", ""), r.get("seed", ""))
            rp, tp = png_pair(res, r)
            if tp is None or not rp.exists() or not tp.exists():
                print("missing PNG for", key, r["idx"])
                continue
            m = metrics(load(rp), load(tp))
            if r.get("mse"):  # whole-image MSE as the attack itself measured it (float tensors, before PNG quantisation)
                m["mse"] = float(r["mse"])
            cells[key][int(r["idx"])] = (r, m)

    def agg(v):
        rows = [x for _, x in sorted(v.items())]
        out = {c: st.mean(m[c] for _, m in rows) for c in ("mse", "ssim", "psnr", "lpips", "mse_fg")}
        out.update({c + "0": rows[0][1][c] for c in ("mse", "ssim", "psnr", "lpips", "mse_fg")})   # image index 0 alone
        out["acc"] = float(rows[-1][0]["acc"]); out["n"] = len(rows)
        return out

    A = {k: agg(v) for k, v in cells.items()}
    base_single = {(k[0], k[2]): a for k, a in A.items() if k[1] == "none" and a["n"] == 1 and k[8] == ""}
    base_multi = {(k[0], k[2]): a for k, a in A.items() if k[1] == "raw" and a["n"] > 1 and k[7] in ("", "0")}

    out = []
    for k, a in sorted(A.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1], kv[0][5], float(kv[0][4]), -(int(kv[0][3]) if kv[0][3] else 0), kv[0][8], kv[0][7])):
        ds, d, sp, kk, w, b, clip, sig, seed = k
        base = (base_multi if a["n"] > 1 else base_single).get((ds, sp))
        cr = "" if d in ("none", "raw") or not kk else ACT_DIM[sp] // int(kk)
        r_fg = a["mse_fg"] / base["mse_fg"] if base else float("nan")
        r_w = a["mse"] / base["mse"] if base else float("nan")
        b0 = base_single.get((ds, sp)); r_fg0 = a["mse_fg0"] / b0["mse_fg"] if b0 else float("nan")
        out.append({"dataset": ds, "split": sp, "method": d, "block": b, "wcc": w, "cr": cr, "k": "" if d in ("none", "raw") else kk,
                    "seed": seed, "dp_sigma": (f"{float(sig):g}" if sig not in ("", None) and float(sig) > 0 else "0"), "n_images": a["n"],
                    "acc": fmt(a["acc"], 2), "mse": fmt(a["mse"], 6), "mse_ratio": fmt(r_w, 4), "ssim": fmt(a["ssim"], 6), "psnr": fmt(a["psnr"], 4),
                    "lpips": fmt(a["lpips"], 6), "mse_fg": fmt(a["mse_fg"], 6), "mse_fg_ratio": fmt(r_fg, 4),
                    "mse_fg_pct": fmt(100 * (r_fg - 1), 2),
                    "mse0": fmt(a["mse0"], 6), "ssim0": fmt(a["ssim0"], 6), "psnr0": fmt(a["psnr0"], 4), "lpips0": fmt(a["lpips0"], 6),
                    "mse_fg0_ratio": fmt(r_fg0, 4)})
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS); wr.writeheader(); wr.writerows(out)
    print(f"wrote {args.out} ({len(out)} cells)")

    # split-2 headline: worst MSE_fg ratio over CR (and lambda_WCC) per method
    hl = defaultdict(list)
    for r in out:
        if r["split"] == 2 and r["n_images"] == 1 and r["seed"] == "" and r["method"] in ("none", "learned_1x1", "fixed_random") and r["mse_fg_ratio"]:
            hl[(r["method"], r["dataset"])].append((float(r["acc"]), float(r["mse_fg_ratio"])))
    hp = Path(args.out).with_name(Path(args.out).stem + "_headline.csv")
    with open(hp, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["method", "dataset", "acc_min", "acc_max", "ratio_worst", "n_cells"])
        for (m, ds), v in sorted(hl.items()):
            wr.writerow([m, ds, f"{min(a for a, _ in v):.2f}", f"{max(a for a, _ in v):.2f}", f"{min(x for _, x in v):.4f}", len(v)])
    print(f"wrote {hp}")


if __name__ == "__main__":
    main()
