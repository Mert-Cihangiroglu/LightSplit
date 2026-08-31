#!/usr/bin/env python3
"""Aggregate the non-IID Dirichlet-alpha sweep and build the
heterogeneity-curve figures.

Grid: CIFAR-10, 4 heads {L1H,L2H}x{SCH,PCH} x 3 seeds x {raw + block_random b1/b2/b4/b8}
x alpha{0.1,0.3,1,3,10} x CR{8,16,32}. block = the locality dial (b1 max utility ...
b8 = global dense R = fixed_random / LightSplit-F). Metric = best global test acc, mean over seeds.

Outputs -> results/noniid-alpha/
  results/ noniid_alpha_master.csv    (head,method,block,seed,alpha,cr,acc)
           noniid_alpha_seedmean.csv  (head,method,block,alpha,cr,mean_acc,std,n)
  figures/ noniid_alpha_curve.{png,pdf}         acc vs alpha, 4 heads x 3 CR, log-y
           noniid_alpha_curve_relraw.{png,pdf}  same, % of raw (relative-to-raw), log-y
"""
import csv, glob, os, re
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/csv")
OUT = "results/noniid-alpha"
FIG, RES = os.path.join(OUT, "figures"), os.path.join(OUT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

HEADS = ["L1H_SCH", "L1H_PCH", "L2H_SCH", "L2H_PCH"]
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]     # common grid (legacy 30..1000 dropped)
CRS = [8, 16, 32]
BLOCKS = ["raw", "b1", "b2", "b4", "b8"]   # raw baseline + the locality dial
BCOL = {"raw": "#000000", "b1": "#0072B2", "b2": "#009E73", "b4": "#E69F00", "b8": "#CC79A7"}

# ---- collect files, keyed by (head, block, seed); prefer _s{seed}, legacy L2H_SCH unsuffixed = seed 0
files = {}
for f in glob.glob(os.path.join(SRC, "noniid_alpha_*_summary.csv")):
    b = os.path.basename(f)
    m = re.match(r"noniid_alpha_(raw|block_random_b(\d))_cifar10_(L[12]H_(?:SCH|PCH))(?:_s(\d))?_summary\.csv$", b)
    if not m:
        continue
    block = "raw" if m[1] == "raw" else f"b{m[2]}"
    head = m[3]; seed = int(m[4]) if m[4] else 0     # unsuffixed -> seed 0 (legacy L2H_SCH)
    key = (head, block, seed)
    # prefer an explicit _s{seed} file over the legacy unsuffixed one
    if key in files and m[4] is None:
        continue
    files[key] = f

# ---- read rows -> data[(head,block,cr,alpha)] = list of accs (one per seed)
data = defaultdict(list)
master = []
for (head, block, seed), f in sorted(files.items()):
    for r in csv.DictReader(open(f)):
        a = float(r["alpha"]);  a = round(a, 3)
        if a not in ALPHAS:      # drop legacy 30/100/... and IID sentinel if any
            continue
        cr = int(float(r["compression_ratio"]))
        acc = float(r["best_global_test_acc"]) * 100
        # raw ignores CR -> tag it to every CR so it draws on each panel
        crs = CRS if block == "raw" else [cr]
        for c in crs:
            data[(head, block, c, a)].append(acc)
        master.append(dict(head=head, method=("raw" if block == "raw" else "block_random"),
                           block=block, seed=seed, alpha=a, cr=cr, acc=round(acc, 3)))

def mean(xs): return sum(xs) / len(xs) if xs else None
def std(xs):
    if len(xs) < 2: return 0.0
    mu = mean(xs); return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

# ---- CSVs
with open(os.path.join(RES, "noniid_alpha_master.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["head", "method", "block", "seed", "alpha", "cr", "acc"])
    w.writeheader(); w.writerows(sorted(master, key=lambda d: (d["head"], d["block"], d["cr"], d["alpha"], d["seed"])))
with open(os.path.join(RES, "noniid_alpha_seedmean.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["head", "block", "cr", "alpha", "n_seeds", "mean_acc", "std_acc"])
    for (head, block, cr, a), accs in sorted(data.items()):
        w.writerow([head, block, cr, a, len(accs), round(mean(accs), 3), round(std(accs), 3)])

# ---- log-y helper (standing rule: all figures log y)
def logify(ax, lo, hi):
    lo = max(1.0, lo * 0.9); hi = hi * 1.1
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ticks = [t for t in [2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

def cell(head, block, cr, a):
    v = data.get((head, block, cr, a)); return mean(v) if v else None

# ---- FIG: acc vs alpha, 4 heads (rows) x 3 CR (cols), line per block ----
for REL in (False, True):
    fig, axes = plt.subplots(len(HEADS), len(CRS), figsize=(4.4 * len(CRS), 3.4 * len(HEADS)), squeeze=False)
    allv = []
    for i, head in enumerate(HEADS):
        for j, cr in enumerate(CRS):
            ax = axes[i][j]; pts = []
            rawline = [cell(head, "raw", cr, a) for a in ALPHAS]
            for block in BLOCKS:
                ys = [cell(head, block, cr, a) for a in ALPHAS]
                if REL:
                    ys = [(100.0 * y / r) if (y is not None and r) else None for y, r in zip(ys, rawline)]
                xs = [a for a, y in zip(ALPHAS, ys) if y is not None]
                yy = [y for y in ys if y is not None]
                if not yy: continue
                pts += yy; allv += yy
                ax.plot(xs, yy, "-o", color=BCOL[block], lw=2, ms=5,
                        label=("fixed_random (b8)" if block == "b8" else block), zorder=(1 if block == "raw" else 2))
            ax.set_xscale("log"); ax.set_xticks(ALPHAS); ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=8)
            if pts: logify(ax, min(pts), max(pts))
            ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(f"CR {cr}", fontsize=11)
            if j == 0: ax.set_ylabel(f"{head}\n" + ("% of raw" if REL else "global test acc %"), fontsize=9)
            if i == len(HEADS) - 1: ax.set_xlabel(r"Dirichlet $\alpha$  (smaller = more non-IID)", fontsize=9)
    axes[0][0].legend(fontsize=8, loc="lower right", ncol=2)
    ttl = ("Non-IID heterogeneity — block_random locality dial (CIFAR-10, mean over 3 seeds)\n"
           + ("% of raw accuracy" if REL else "global test acc") +
           " vs Dirichlet α; line per block (b1 max-utility → b8=fixed_random); log y")
    fig.suptitle(ttl, y=1.002, fontsize=11)
    fig.tight_layout()
    sfx = "_relraw" if REL else ""
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"noniid_alpha_curve{sfx}.{e}"), dpi=140, bbox_inches="tight")
    plt.close(fig)

print(f"wrote {len(files)} cells -> {OUT}")
print(f"  master rows: {len(master)} ; seed-mean cells: {len(data)}")
# quick sanity: heterogeneity gradient + locality dial at CR8, L2H_SCH
print("\nsanity L2H_SCH CR8 (acc %):  alpha ->", ALPHAS)
for block in BLOCKS:
    print(f"  {block:>4}: " + " ".join(f"{(cell('L2H_SCH', block, 8, a) or float('nan')):5.1f}" for a in ALPHAS))
