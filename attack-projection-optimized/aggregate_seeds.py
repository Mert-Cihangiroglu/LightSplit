#!/usr/bin/env python3
"""Cross-seed aggregation: pool the wb_block known-R grid across seeds and emit
mean+/-std per cell, so every published number carries an error bar (n=3: seeds 42,43,44).

Directory convention (block from dir name, lambda from dir name, seed from suffix):
  seed 42 (original grid):  results/wb_block_b{B}_l{L}[_cr16]
  reseed:                   results/wb_block_b{B}_l{L}[_cr16]_s{SEED}
CR is read from the row's `cr` column (exp3) / `cr` (exp4), NOT from the dir, so the
CR8/CR32 dirs contribute two CR groups each and the _cr16 dirs contribute CR16.

Per (block, CR, lambda) cell we collect across seeds:
  known_ssim   (decoder + analytic lift-back z R^T)
  unknown_ssim (decoder from z alone)
  delta = known_ssim - unknown_ssim   (the headline "pinv buys nothing" quantity)
  frac_lost    (null-space energy discarded, exp4)
and report mean, std, n. Also prints the grand headline: mean/std of delta over all cells.

Outputs: wb_block_ssim_grid_seeded.csv
Usage: python aggregate_seeds.py [--seeds 42 43 44] [--out <path>]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_RESULTS = RESULTS

# wb_block_b{B}_l{L}  optional _cr16  optional _s{SEED}
DIR_RE = re.compile(r"^wb_block_b(\d+)_l([\d.]+?)(_cr16)?(?:_s(\d+))?$")


def parse_dir(name):
    m = DIR_RE.match(name)
    if not m:
        return None
    block = int(m.group(1))
    lam = float(m.group(2))
    seed = int(m.group(4)) if m.group(4) else 42   # no suffix == the original seed-42 grid
    return block, lam, seed


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None   # drop NaN
    except (TypeError, ValueError):
        return None


def main(seeds, out):
    # cell -> {known:[], unknown:[], delta:[], frac_lost:[]}, keyed (block, cr, lam)
    cells = defaultdict(lambda: defaultdict(list))
    seen_seeds = set()
    ndirs = 0
    for d in sorted(RESULTS.glob("wb_block_b*")):
        if not d.is_dir():
            continue
        meta = parse_dir(d.name)
        if meta is None:
            continue
        block, lam, seed = meta
        if seeds and seed not in seeds:
            continue
        summ = d / "summary.csv"
        if not summ.exists():
            continue
        ndirs += 1
        seen_seeds.add(seed)
        with open(summ) as f:
            for row in csv.DictReader(f):
                exp = row.get("experiment", "")
                cr = fnum(row.get("cr"))
                if cr is None:
                    continue
                key = (block, int(cr), lam)
                if exp == "exp3_known_vs_unknown":
                    k = fnum(row.get("known_ssim"))
                    u = fnum(row.get("unknown_ssim"))
                    if k is not None:
                        cells[key]["known"].append((seed, k))
                    if u is not None:
                        cells[key]["unknown"].append((seed, u))
                    if k is not None and u is not None:
                        cells[key]["delta"].append((seed, k - u))
                elif exp == "exp4_nullspace":
                    fl = fnum(row.get("frac_lost_mean"))
                    if fl is not None:
                        cells[key]["frac_lost"].append((seed, fl))

    def ms(pairs):
        vals = [v for _, v in pairs]
        if not vals:
            return None, None, 0
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0), len(vals)

    rows = []
    all_deltas = []
    for (block, cr, lam) in sorted(cells):
        c = cells[(block, cr, lam)]
        km, ks, kn = ms(c["known"])
        um, us, un = ms(c["unknown"])
        dm, ds, dn = ms(c["delta"])
        fm, fs, fn = ms(c["frac_lost"])
        if dm is not None:
            all_deltas.append(dm)
        rows.append(dict(block=block, cr=cr, wcc_lambda=lam,
                         known_ssim_mean=km, known_ssim_std=ks,
                         unknown_ssim_mean=um, unknown_ssim_std=us,
                         delta_mean=dm, delta_std=ds,
                         frac_lost_mean=fm, frac_lost_std=fs,
                         n_seeds=max(kn, un, dn, fn)))

    out = Path(out) if out else (DEFAULT_RESULTS / "wb_block_ssim_grid_seeded.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["block", "cr", "wcc_lambda",
            "known_ssim_mean", "known_ssim_std",
            "unknown_ssim_mean", "unknown_ssim_std",
            "delta_mean", "delta_std",
            "frac_lost_mean", "frac_lost_std", "n_seeds"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})

    print(f"seeds present: {sorted(seen_seeds)}  ({ndirs} cell-dirs read)")
    print(f"cells: {len(rows)}   wrote {out}")
    if all_deltas:
        gm = mean(all_deltas); gs = pstdev(all_deltas) if len(all_deltas) > 1 else 0.0
        print(f"\nHEADLINE  mean(known_ssim - unknown_ssim) over {len(all_deltas)} cells "
              f"= {gm:+.4f} +/- {gs:.4f}   max = {max(all_deltas):+.4f}")
        print("(pinv/lift-back feature buys ~nothing; the defense is rank deficiency, not R-secrecy.)")
    # per-seed coverage sanity
    nfull = sum(1 for r in rows if r["n_seeds"] >= len(seen_seeds))
    print(f"cells with full n={len(seen_seeds)} coverage: {nfull}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds (default: all found, i.e. 42,43,44)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(set(args.seeds) if args.seeds else None, args.out)
