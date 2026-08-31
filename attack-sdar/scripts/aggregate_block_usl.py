#!/usr/bin/env python3
"""Aggregate USL SDAR reconstruction SSIM for block_random (+ topk_sparse) from the run logs.

Each run log prints 'Average SSIM ...' and 'Average label accuracy ...'. In USL the label
accuracy is the victim-health signal: WCC>0 often collapses the victim, and a collapsed
victim's SSIM is not a trustworthy privacy number. We HEALTH-GATE (default victim acc >=
0.40) and average SSIM over the healthy seeds per (CR, block, WCC).

Usage:
  python aggregate_block_usl.py --level 7                # all seeds found
  python aggregate_block_usl.py --level 7 --gate 0.40
Outputs a table to stdout + block_usl_l<level>_ssim.csv
"""
import argparse, csv, glob, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--level", type=int, default=7)
ap.add_argument("--gate", type=float, default=0.40, help="min victim label-acc to trust a seed")
ap.add_argument("--logdir", default="reseed_logs")
ap.add_argument("--seed", type=int, default=None, help="restrict to a single seed (e.g. 0)")
args = ap.parse_args()

SSIM_RE = re.compile(r"Average SSIM over all client's images:\s*(\d+\.\d+)")
ACC_RE = re.compile(r"Average label accuracy over all client's images:\s*(\d+\.\d+)")
# usl_l7_blockrandom_cr8_b2[_wcc0.01]_run0.log
NAME_RE = re.compile(r"blockrandom_cr(\d+)_b(\d+)(?:_wcc([0-9.]+))?_run(\d+)\.log$")


def parse_log(path):
    ssim = acc = None
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return None
    ms, ma = SSIM_RE.findall(txt), ACC_RE.findall(txt)
    if ms:
        ssim = float(ms[-1])
    if ma:
        acc = float(ma[-1])
    return ssim, acc


# cell -> {seed: (ssim, acc)}
cells = defaultdict(dict)
for f in glob.glob(os.path.join(args.logdir, f"usl_l{args.level}_blockrandom_*_run*.log")):
    m = NAME_RE.search(os.path.basename(f))
    if not m:
        continue
    cr, block, wcc, seed = int(m.group(1)), int(m.group(2)), float(m.group(3) or 0), int(m.group(4))
    if args.seed is not None and seed != args.seed:
        continue
    r = parse_log(f)
    if r and r[0] is not None:
        cells[(cr, block, wcc)][seed] = r

rows = []
for (cr, block, wcc), seeds in sorted(cells.items()):
    healthy = [(s, ss) for s, (ss, ac) in seeds.items() if ac is not None and ac >= args.gate]
    all_seeds = sorted(seeds)
    n_h = len(healthy)
    mean_ssim = sum(ss for _, ss in healthy) / n_h if n_h else float("nan")
    rows.append({
        "CR": cr, "block": block, "wcc": wcc,
        "n_seeds": len(all_seeds), "n_healthy": n_h,
        "mean_ssim_healthy": round(mean_ssim, 4) if n_h else "",
        "seed_ssims": ";".join(f"{s}:{seeds[s][0]:.3f}(acc{(seeds[s][1] or 0):.2f})" for s in all_seeds),
    })

out = f"block_usl_l{args.level}_ssim.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"level {args.level} | gate victim-acc>={args.gate} | {len(rows)} cells | -> {out}\n")
print(f"{'CR':>3} {'blk':>3} {'wcc':>6} {'seeds':>5} {'healthy':>7} {'mean_SSIM':>9}   seeds(ssim/acc)")
for r in rows:
    print(f"{r['CR']:>3} {r['block']:>3} {r['wcc']:>6} {r['n_seeds']:>5} {r['n_healthy']:>7} "
          f"{str(r['mean_ssim_healthy']):>9}   {r['seed_ssims']}")
# quick read: projection-only (wcc0) locality dial per CR
print("\nlocality dial (wcc=0, projection only), mean healthy SSIM:")
for cr in (8, 16, 32):
    line = "  ".join(f"b{b}={next((r['mean_ssim_healthy'] for r in rows if r['CR']==cr and r['block']==b and r['wcc']==0), '-')}"
                      for b in (1, 2, 4))
    print(f"  CR{cr}: {line}")
