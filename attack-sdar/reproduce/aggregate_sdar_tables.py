#!/usr/bin/env python3
"""Aggregate SDAR run logs into the paper's SDAR tables.

  python aggregate_sdar_tables.py --logs DIR [--logs DIR ...] --scripts <attack-sdar>/scripts \
      --out-dir results [--map cells.json]

Needs torch + lpips (run it with a PyTorch environment; the runs themselves use TensorFlow).

Every *.log under the --logs directories is parsed: protocol/level from the run header,
defense from the "Compression defense:" / "DP defense:" lines, final MSE / SSIM / attacker
label accuracy from the summary lines, and the 20-image reconstruction grid from the
"History saved to" path (resolved against --scripts). PSNR = -10 log10(MSE); LPIPS_20 is
LPIPS(alex) between the run's grid and the originals grid of the same level, averaged over
the 20 images.

Cell protocols, as in the paper:
  raw / learned_1x1 / fixed_random   one attacker run per cell.
  topk_sparse                        mean over runs whose ATTACKER converged
                                     (label accuracy >= 0.40; a failed attacker scores
                                     near-zero SSIM regardless of the defense).
  DP rows (raw+DP, fixed_random CR8+DP)   mean over runs; a run is dropped when its victim
                                     collapsed or its simulator collapsed (fg_acc <= 0.5 or
                                     e_dis_loss <= 0.12 in the saved history, when present).

--map (optional) pins cells to explicit log files: {"<setting>_l<level>_<cell>": [paths]}.
Used to validate the aggregation against the original run set; without it, logs are
matched by their parsed configuration.

Outputs in --out-dir:
  tab_sdar_all_split_points.csv  level, method, cr, wcc, {usl,vsl}_{ssim,psnr,lpips}
  tab_sdar_compact.csv           USL level-4 CR8: mse, ssim, psnr, lpips, delta_raw
  tab_dp_sdar_privacy.csv        the DP SSIM cells at both cuts
  tab_recon_main_sdar.csv        the SDAR pair of the consolidated privacy table
                                 (USL level 4, CR 8: LPIPS / SSIM percent change vs Raw)
"""
import argparse, csv, glob, json, math, os, re
from pathlib import Path
import numpy as np
from PIL import Image

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
HDR = re.compile(r"Start (\w+)/\w+ experiments on (\w+) at l(\d+)")
COMP = re.compile(r"^Compression defense: (.*)$", re.M)
C_MODE = re.compile(r"mode=(\w+)")
C_CR = re.compile(r"compression_ratio=" + NUM)
C_WCC = re.compile(r"wcc_lambda=" + NUM)
DP = re.compile(r"DP defense: dp_clip_norm=" + NUM + r", dp_sigma=" + NUM)
MSE = re.compile(r"Average MSE over all client's images:\s*" + NUM)
SSIM = re.compile(r"Average SSIM over all client's images:\s*" + NUM)
ACC = re.compile(r"Average label accuracy over all client's images:\s*" + NUM)
NPY = re.compile(r"History saved to (\S+)")
GATE = 0.40          # attacker convergence (topk_sparse)
FG_MIN, EDIS_MIN = 0.50, 0.12   # victim / simulator health (DP rows)


def parse_log(path, scripts):
    t = open(path, errors="ignore").read()
    h = HDR.search(t)
    m, s = MSE.findall(t), SSIM.findall(t)
    if not h or not s:
        return None
    comp = COMP.search(t)
    mode, cr, wcc = "raw", 0, 0.0
    if comp and "None" not in comp.group(1):
        c = comp.group(1)
        mm = C_MODE.search(c)
        mode = mm.group(1) if mm else "raw"
        mc, mw = C_CR.search(c), C_WCC.search(c)
        cr = int(float(mc.group(1))) if mc else 0
        wcc = float(mw.group(1)) if mw else 0.0
    dp = DP.search(t)
    acc = ACC.findall(t)
    npy = NPY.findall(t)
    grid = hist = None
    if npy:
        rel = Path(npy[-1])
        png_name = rel.name.replace(".npy", ".png")
        roots = [Path(rel.anchor)] if rel.is_absolute() else             [Path(scripts), Path(path).resolve().parent.parent]
        rel = rel.relative_to(rel.anchor) if rel.is_absolute() else rel
        for root in roots:
            # the run writes the history under .../output/ and the grid under .../fig/,
            # relative to the scripts/ directory it ran from
            for base in (root / rel.parent, root / rel.parent / "output"):
                if (base / rel.name).exists() or (base.parent / "fig" / png_name).exists()                         or (base / "fig" / png_name).exists():
                    hist = base / rel.name
                    fig = base.parent / "fig" if base.name == "output" else base / "fig"
                    grid = fig / png_name
                    break
            if hist is not None:
                break
    return dict(setting=h.group(1), level=int(h.group(3)), mode=mode, cr=cr, wcc=wcc,
                dp_sigma=float(dp.group(2)) if dp else 0.0,
                mse=float(m[-1].rstrip(".")) if m else None,
                ssim=float(s[-1].rstrip(".")), acc=float(acc[-1].rstrip(".")) if acc else None,
                grid=grid, hist=hist, log=path)


def strip_grid(path, n=20, tile=64):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return np.stack([np.asarray(img.crop((round(i * w / n), 0, round((i + 1) * w / n), h))
                                .resize((tile, tile), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
                     for i in range(n)])


class Lpips:
    def __init__(self):
        import torch, lpips
        self.torch, self.model = torch, lpips.LPIPS(net="alex", verbose=False).eval()
        self.orig = {}

    def against(self, level, originals, grid_png):
        if level not in self.orig:
            self.orig[level] = strip_grid(originals[level])
        a = self.torch.from_numpy(self.orig[level] * 2 - 1).permute(0, 3, 1, 2)
        b = self.torch.from_numpy(strip_grid(grid_png) * 2 - 1).permute(0, 3, 1, 2)
        with self.torch.no_grad():
            return float(self.model(a, b).view(-1).mean())


def health(hist):
    """(fg_acc, e_dis_loss) tails from the saved history, or (None, None)."""
    if hist is None or not Path(hist).exists():
        return None, None
    try:
        d = np.load(hist, allow_pickle=True).item()
    except Exception:
        return None, None
    def tail(k, n=500):
        a = d.get(k)
        return float(np.mean(np.asarray(a)[-n:])) if a is not None else None
    return tail("fg_acc"), tail("e_dis_loss")


def cell_key(r):
    if r["dp_sigma"]:
        base = "rawdp" if r["mode"] == "raw" else f"cr{r['cr']}dp"
        return (r["setting"], r["level"], base, 0.0)
    if r["mode"] == "raw":
        return (r["setting"], r["level"], "raw", 0.0)
    return (r["setting"], r["level"], f"{r['mode']}_cr{r['cr']}", r["wcc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", action="append", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--map", help="JSON {cell tag: [log files]} pinning the run set per cell")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.map:
        pinned = json.load(open(args.map))
        files = {}
        def norm(e):
            return (e, None) if isinstance(e, str) else (e[0], e[1])
        cells = {}
        for tag, entries in pinned.items():
            rs = []
            for e in entries:
                logp, gridp = norm(e)
                r = parse_log(logp, args.scripts)
                if r is None:
                    continue
                if gridp:
                    r["grid"] = Path(gridp)
                rs.append(r)
            cells[tag] = rs
    else:
        cells = {}
        for d in args.logs:
            for p in sorted(glob.glob(os.path.join(d, "**", "*.log"), recursive=True)):
                r = parse_log(p, args.scripts)
                if r is None:
                    continue
                s, L, base, wcc = cell_key(r)
                w = f"_wcc{wcc:g}" if base.startswith("fixed_random") else ""
                cells.setdefault(f"{s}_l{L}_{base}{w}", []).append(r)

    originals = {L: Path(args.scripts) / "sdar_results/vsl/cifar10/fig" / f"resnet_l{L}_run1_originals.png"
                 for L in (4, 7)}
    lp = Lpips()

    def score(runs, protocol):
        """-> dict(mse, ssim, psnr, lpips) or None"""
        if protocol == "single":
            runs = runs[:1]
        elif protocol == "topk":
            # VSL logs do not print the attacker label accuracy; keep those runs
            runs = [r for r in runs if r["acc"] is None or r["acc"] >= GATE]
        elif protocol == "dp":
            kept = []
            for r in runs:
                fg, ed = health(r["hist"])
                if fg is not None and (fg <= FG_MIN or (ed is not None and ed <= EDIS_MIN)):
                    print(f"  [drop unhealthy] {os.path.basename(r['log'])} fg={fg:.3f}")
                    continue
                kept.append(r)
            runs = kept
        if not runs:
            return None
        vals = dict(mse=[], ssim=[], psnr=[], lpips=[])
        for r in runs:
            vals["ssim"].append(r["ssim"])
            if r["mse"]:
                vals["mse"].append(r["mse"])
                vals["psnr"].append(-10 * math.log10(r["mse"]))
            if r["grid"] is not None and r["grid"].exists():
                vals["lpips"].append(lp.against(r["level"], originals, r["grid"]))
        return {k: (sum(v) / len(v) if v else None) for k, v in vals.items()} | {"n": len(runs)}

    def get(setting, level, base, wcc=None, protocol="single"):
        tag = f"{setting}_l{level}_{base}" + (f"_wcc{wcc:g}" if wcc is not None else "")
        runs = cells.get(tag)
        return score(sorted(runs, key=lambda r: r["log"]), protocol) if runs else None

    # ---- merged all-split-points table ----
    ROWS = [("Raw", "raw", None, "single"), ("Raw+DP", "rawdp", None, "dp")]
    for cr in (8, 16, 32):
        ROWS.append(("Learned1x1", f"learned_1x1_cr{cr}", None, "single"))
    for w in (0.0, 0.01, 0.1):
        for cr in (8, 16, 32):
            ROWS.append(("LightSplit-F", f"fixed_random_cr{cr}", w, "single"))
    for cr in (8, 16, 32):
        ROWS.append(("topk_sparse", f"topk_sparse_cr{cr}", None, "topk"))
    ROWS.append(("LightSplit-F+DP", "cr8dp", None, "dp"))
    DPWCC = {"cr8dp": "0"}

    fmt = lambda v, nd: "" if v is None else f"{v:.{nd + 2}f}"
    with open(os.path.join(args.out_dir, "tab_sdar_all_split_points.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "method", "cr", "wcc",
                    "usl_ssim", "usl_psnr", "usl_lpips", "vsl_ssim", "vsl_psnr", "vsl_lpips"])
        for level in (7, 4):
            for name, base, wcc, proto in ROWS:
                cr = (base.split("cr")[-1].split("dp")[0] if "cr" in base else "")
                row = [level, name, 8 if base == "cr8dp" else cr,
                       DPWCC.get(base, "" if wcc is None else f"{wcc:g}")]
                for setting in ("usl", "vsl"):
                    m = get(setting, level, base, wcc, proto)
                    row += [fmt(m and m["ssim"], 3), fmt(m and m["psnr"], 2), fmt(m and m["lpips"], 4)]
                w.writerow(row)

    # ---- compact table: USL level 4, CR 8 ----
    with open(os.path.join(args.out_dir, "tab_sdar_compact.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "wcc", "mse", "ssim", "psnr", "lpips", "delta_raw"])
        raw = get("usl", 4, "raw")
        for name, base, wcc, proto in [("Raw", "raw", None, "single"),
                                       ("Learned1x1", "learned_1x1_cr8", None, "single"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.0, "single"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.01, "single"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.1, "single")]:
            m = get("usl", 4, base, wcc, proto)
            if m is None:
                w.writerow([name, "" if wcc is None else f"{wcc:g}"] + [""] * 5)
                continue
            dr = m["lpips"] / raw["lpips"] if (raw and raw["lpips"] and m["lpips"]) else None
            w.writerow([name, "" if wcc is None else f"{wcc:g}", fmt(m["mse"], 4), fmt(m["ssim"], 4),
                        fmt(m["psnr"], 2), fmt(m["lpips"], 4), fmt(dr, 2)])

    # ---- DP table (SSIM at both cuts; level 7 = 4096-d, level 4 = 8192-d) ----
    with open(os.path.join(args.out_dir, "tab_dp_sdar_privacy.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "l7_ssim", "l4_ssim"])
        for name, base, wcc, proto in [("Raw", "raw", None, "single"), ("Raw+DP", "rawdp", None, "dp"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.0, "single"),
                                       ("LightSplit-F+DP", "cr8dp", None, "dp")]:
            row = [name]
            for level in (7, 4):
                m = get("usl", level, base, wcc, proto)
                row.append(fmt(m and m["ssim"], 3))
            w.writerow(row)
    # ---- consolidated privacy table (Table III) SDAR pair: USL level 4, CR 8 ----
    with open(os.path.join(args.out_dir, "tab_recon_main_sdar.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "wcc", "lpips", "ssim", "lpips_gain_pct", "ssim_gain_pct"])
        raw = get("usl", 4, "raw")
        for name, base, wcc, proto in [("Raw", "raw", None, "single"),
                                       ("Raw+DP", "rawdp", None, "dp"),
                                       ("Learned1x1", "learned_1x1_cr8", None, "single"),
                                       ("topk_sparse", "topk_sparse_cr8", None, "topk"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.0, "single"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.001, "single"),
                                       ("LightSplit-F", "fixed_random_cr8", 0.1, "single"),
                                       ("LightSplit-F+DP", "cr8dp", None, "dp")]:
            m = get("usl", 4, base, wcc, proto)
            row = [name, "" if wcc is None else f"{wcc:g}"]
            if m is None or raw is None:
                w.writerow(row + [""] * 4)
                continue
            row += [fmt(m["lpips"], 4), fmt(m["ssim"], 3)]
            if name == "Raw":
                row += ["", ""]
            else:
                lg = "" if (m["lpips"] is None or not raw["lpips"]) else                     f"{(m['lpips'] - raw['lpips']) / raw['lpips'] * 100:+.1f}"
                sg = "" if m["ssim"] is None else f"{(m['ssim'] - raw['ssim']) / raw['ssim'] * 100:+.1f}"
                row += [lg, sg]
            w.writerow(row)
    print(f"[aggregate_sdar_tables] {len(cells)} cells -> {args.out_dir}")


if __name__ == "__main__":
    main()
