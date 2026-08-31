#!/usr/bin/env python
"""Aggregate FORA runs into the paper's FORA tables.

A cell is one configuration run with several attacker seeds. Each run is scored on the 32-image
reconstruction grid it saves at its last evaluation (`recon_pics/<tag>/2/{pseudo,target}/recon_<iter>.png`:
rows alternate reconstruction / ground truth, 8 images per row, 32 pairs) with piq's SSIM, PSNR
and LPIPS (VGG). `pseudo` is FORA's deployable surrogate-trained decoder, `target` the white-box
reference decoder. A cell is the mean over its runs; the pseudo decoder does not converge in
every run, so the pseudo columns average the converged runs only (a run counts as collapsed
when its pseudo SSIM is below -0.3 and its PSNR below 9 dB while another run of the cell
reconstructs; a cell whose runs all collapsed gets empty pseudo cells). The reference decoder
converged in every run and averages all of them. Victim accuracy is the last target-classifier
evaluation in each run's log (the second-to-last "Accuracy: a/b" line), averaged over runs.

Output CSV, one row per cell: method, block, wcc, k, cr, dp_sigma, n_runs, n_pseudo, pseudo_ssim,
pseudo_psnr, pseudo_lpips, pseudo_lpips_pct and pseudo_ssim_pct (change vs the Raw pseudo cell,
the main tables' FORA columns), ref_ssim, ref_psnr, ref_lpips, ref_lpips_ratio (Delta_Raw of the
full table), victim_acc, runs.

  python aggregate_fora.py --root results/reproduce/tab_fora --out results/reproduce/tab_fora.csv

--root may be repeated; --tag-map JSON renames runs saved under other names ({"<cell>_s<seed>": "<tag>"}).
"""
import argparse, csv, json, re, statistics as st
from pathlib import Path
import numpy as np, torch, piq
from PIL import Image

IMG, PAD, NPAIRS = 32, 2, 32
CR_OF_K = {"2048": 8, "1024": 16, "512": 32, "8": 8, "4": 16, "2": 32}
S3, S2 = "0 1 2", "0 1"
# cell -> (method, block, wcc, k, dp_sigma, seeds); the driver runs exactly these
CELLS = [
    ("raw",                      "none",         "", "0",     "",     "0",    S3),
    ("rawdp",                    "none",         "", "0",     "",     "0.02", S3),
    ("learned1x1_k8",            "learned_1x1",  "", "0",     "8",    "0",    "0 1 2 3"),
    ("learned1x1_k4",            "learned_1x1",  "", "0",     "4",    "0",    S3),
    ("learned1x1_k2",            "learned_1x1",  "", "0",     "2",    "0",    S3),
    ("fixedrandom_k2048_wcc0",   "fixed_random", "", "0",     "2048", "0",    S3),
    ("fixedrandom_k1024_wcc0",   "fixed_random", "", "0",     "1024", "0",    S3),
    ("fixedrandom_k512_wcc0",    "fixed_random", "", "0",     "512",  "0",    S3),
    ("fixedrandom_k2048_wcc0.001", "fixed_random", "", "0.001", "2048", "0",  S3),
    ("fixedrandom_k2048_wcc0.01", "fixed_random", "", "0.01",  "2048", "0",   "0"),
    ("fixedrandom_k1024_wcc0.01", "fixed_random", "", "0.01",  "1024", "0",   "0"),
    ("fixedrandom_k512_wcc0.01",  "fixed_random", "", "0.01",  "512",  "0",   "0"),
    ("fixedrandom_k2048_wcc0.1",  "fixed_random", "", "0.1",   "2048", "0",   S3),
    ("fixedrandom_k1024_wcc0.1",  "fixed_random", "", "0.1",   "1024", "0",   "0"),
    ("fixedrandom_k512_wcc0.1",   "fixed_random", "", "0.1",   "512",  "0",   "0"),
    ("topksparse_k2048",         "topk_sparse",  "", "0",     "2048", "0",    S3),
    ("topksparse_k1024",         "topk_sparse",  "", "0",     "1024", "0",    S2),
    ("topksparse_k512",          "topk_sparse",  "", "0",     "512",  "0",    S2),
    ("dp_fixedrandom_k2048",     "fixed_random", "", "0",     "2048", "0.02", S3),
    ("dp_fixedrandom_k1024",     "fixed_random", "", "0",     "1024", "0.02", S3),
    ("dp_fixedrandom_k512",      "fixed_random", "", "0",     "512",  "0.02", S3),
] + [(f"blockrandom_b{b}_k{k}", "block_random", str(b), "0", str(k), "0", "0") for b in (1, 2, 4) for k in (2048, 1024, 512)]

_lpips = None


def grid_pairs(png):
    img = np.array(Image.open(png).convert("RGB")); cell = IMG + PAD
    rec, org = [], []
    for pi in range(4):
        for col in range(8):
            x0 = PAD + col * cell; yr = PAD + (2 * pi) * cell; yo = PAD + (2 * pi + 1) * cell
            r, o = img[yr:yr + IMG, x0:x0 + IMG], img[yo:yo + IMG, x0:x0 + IMG]
            if r.shape == (IMG, IMG, 3) and o.shape == (IMG, IMG, 3):
                rec.append(r); org.append(o)
    t = lambda L: torch.stack([torch.from_numpy(a).permute(2, 0, 1).float() / 255. for a in L])
    return t(org)[:NPAIRS], t(rec)[:NPAIRS]


def score(png):
    global _lpips
    if _lpips is None:
        _lpips = piq.LPIPS(reduction="mean")
    o, r = grid_pairs(png)
    with torch.no_grad():
        return dict(ssim=piq.ssim(o, r, data_range=1.0, reduction="mean").item(),
                    psnr=piq.psnr(o, r, data_range=1.0, reduction="mean").item(),
                    lpips=_lpips(o, r).item())


def last_grid(d):
    its = [int(m.group(1)) for f in d.glob("recon_*.png") if (m := re.match(r"recon_(\d+)\.png$", f.name))]
    return (max(its), d / f"recon_{max(its)}.png") if its else (None, None)


def find_run(roots, tag, prefix):
    short = tag[len(prefix):] if tag.startswith(prefix) else tag
    for root in roots:
        base = root / "recon_pics" / tag / "2"
        if base.is_dir():
            logs = [p for p in (root / "logs" / f"{tag}_full.log", root / f"{tag}_full.log", root / f"{short}_full.log") if p.exists()]
            return base, (logs[0] if logs else None)
    return None, None


def victim_acc(log):
    if log is None:
        return float("nan")
    accs = re.findall(r"Accuracy:\s*(\d+)/(\d+)", log.read_text(errors="ignore"))
    return 100.0 * int(accs[-2][0]) / int(accs[-2][1]) if len(accs) >= 2 else float("nan")


def collapsed(p):
    return p["ssim"] < -0.3 and p["psnr"] < 9


def mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="run directory (recon_pics/ and logs/); repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag-map", help="JSON {<cell>_s<seed>: run tag without prefix} for runs saved under other names")
    ap.add_argument("--prefix", default="cifar_level2_")
    a = ap.parse_args()
    roots = [Path(r) for r in a.root]; tmap = json.load(open(a.tag_map)) if a.tag_map else {}
    rows = []
    for cell, method, block, wcc, k, sig, seeds in CELLS:
        runs = []
        for s in seeds.split():
            name = f"{cell}_s{s}"; tag = a.prefix + tmap.get(name, name)
            tag = tag[:-1] if tag.endswith("_") else tag
            base, log = find_run(roots, tag, a.prefix)
            if base is None:
                print("no run for", tag); continue
            it, pp = last_grid(base / "pseudo"); _, tp = last_grid(base / "target")
            if pp is None or tp is None:
                print("no reconstruction grid for", tag); continue
            runs.append(dict(tag=tag, iter=it, ps=score(pp), tg=score(tp), acc=victim_acc(log)))
        if not runs:
            continue
        conv = [r for r in runs if not collapsed(r["ps"])]
        row = dict(method=method, block=block, wcc=wcc, k=k, cr=CR_OF_K.get(k, ""), dp_sigma=sig, n_runs=len(runs), n_pseudo=len(conv),
                   pseudo_ssim=mean(r["ps"]["ssim"] for r in conv), pseudo_psnr=mean(r["ps"]["psnr"] for r in conv), pseudo_lpips=mean(r["ps"]["lpips"] for r in conv),
                   ref_ssim=mean(r["tg"]["ssim"] for r in runs), ref_psnr=mean(r["tg"]["psnr"] for r in runs), ref_lpips=mean(r["tg"]["lpips"] for r in runs),
                   victim_acc=mean(r["acc"] for r in runs), runs=" ".join(r["tag"] for r in runs))
        rows.append(row)
    raw = next((r for r in rows if r["method"] == "none" and r["dp_sigma"] == "0"), None)
    for r in rows:
        r["pseudo_lpips_pct"] = 100 * (r["pseudo_lpips"] / raw["pseudo_lpips"] - 1) if raw else float("nan")
        r["pseudo_ssim_pct"] = 100 * (r["pseudo_ssim"] / raw["pseudo_ssim"] - 1) if raw else float("nan")
        r["ref_lpips_ratio"] = r["ref_lpips"] / raw["ref_lpips"] if raw else float("nan")
    cols = ["method", "block", "wcc", "k", "cr", "dp_sigma", "n_runs", "n_pseudo", "pseudo_ssim", "pseudo_psnr", "pseudo_lpips",
            "pseudo_lpips_pct", "pseudo_ssim_pct", "ref_ssim", "ref_psnr", "ref_lpips", "ref_lpips_ratio", "victim_acc", "runs"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: ("" if isinstance(r[c], float) and r[c] != r[c] else f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in cols})
    print(f"wrote {a.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
