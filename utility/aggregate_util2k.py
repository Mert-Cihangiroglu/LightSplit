#!/usr/bin/env python3
"""Aggregate the bgq block-utility GRID (util2k_*_summary.csv) into one table.

Pivots the two server-liftback modes side by side:
  block_random_ll (server MLP liftback)  vs  block_random (transpose->backbone)
so the ll-minus-transpose delta is visible per cell. Re-run anytime; resume-safe.
Key per cell = (dataset, dist, head, block, CR). Head+block come from the filename
(util2k_<dataset>_<HEAD>_b<B>_summary.csv); mode/CR/alpha/acc from the row.
Writes a tidy master CSV next to the inputs.
"""
import csv, glob, os, sys

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "csv")
OUT = os.path.join(CSV_DIR, "util2k_master.csv")
# nominal grid size: 4ds x 2dist x 4head x 3CR x 2mode x blocks(b1,b2,b4; gtsrb no b4)
NOMINAL = (3*3 + 2) * 4 * 2 * 3 * 2  # =528 (per-dataset: cifar10/100/fmnist 3 blk, gtsrb 2)

def dist_label(alpha):
    try: a = float(alpha)
    except (TypeError, ValueError): return str(alpha)
    return "IID" if a >= 1e6 else f"niid{a:g}"

def parse_name(fn):
    base = os.path.basename(fn)[len("util2k_"):-len("_summary.csv")]
    p = base.split("_")                      # [ds, Lx, xCH, bN]
    return p[0], f"{p[1]}_{p[2]}", p[3][1:]  # dataset, head, block

rows = {}   # key -> {"ll": acc, "tr": acc}
for fn in sorted(glob.glob(os.path.join(CSV_DIR, "util2k_*_summary.csv"))):
    if os.path.basename(fn) == "util2k_master.csv":
        continue
    ds, head, blk = parse_name(fn)
    with open(fn) as fh:
        for r in csv.DictReader(fh):
            try: cr = int(float(r["compression_ratio"]))
            except (KeyError, ValueError): continue
            key = (ds, dist_label(r["alpha"]), head, f"b{blk}", cr)
            slot = "ll" if r["mode"] == "block_random_ll" else "tr"
            try: acc = float(r["best_global_test_acc"])
            except (KeyError, ValueError): continue
            rows.setdefault(key, {})[slot] = acc

# table
order = sorted(rows)
print(f"{'dataset':8} {'dist':6} {'head':8} {'blk':4} {'CR':>3} "
      f"{'ll(MLP)':>8} {'tr(bbone)':>9} {'ll-tr':>7}")
print("-" * 60)
last = None
both = 0
for key in order:
    ds, dist, head, blk, cr = key
    if last and last != (ds, dist): print()
    last = (ds, dist)
    ll = rows[key].get("ll"); tr = rows[key].get("tr")
    d = f"{ll-tr:+.4f}" if (ll is not None and tr is not None) else ""
    if ll is not None and tr is not None: both += 1
    print(f"{ds:8} {dist:6} {head:8} {blk:4} {cr:>3} "
          f"{('%.4f'%ll) if ll is not None else '  -  ':>8} "
          f"{('%.4f'%tr) if tr is not None else '  -  ':>9} {d:>7}")

# master csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset","dist","head","block","CR","acc_ll_mlp","acc_tr_backbone","delta_ll_minus_tr"])
    for key in order:
        ds, dist, head, blk, cr = key
        ll = rows[key].get("ll"); tr = rows[key].get("tr")
        d = (ll-tr) if (ll is not None and tr is not None) else ""
        w.writerow([ds, dist, head, blk, cr,
                    "" if ll is None else ll, "" if tr is None else tr, d])

ncells = len(rows)
nfull = sum(1 for v in rows.values() if "ll" in v and "tr" in v)
print(f"\ncells seen: {ncells}  (both modes done: {nfull})  |  nominal grid: {NOMINAL}")
print(f"mode-rows: ll={sum('ll' in v for v in rows.values())}  tr={sum('tr' in v for v in rows.values())}")
print(f"master -> {OUT}")
