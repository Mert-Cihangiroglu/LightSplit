#!/usr/bin/env python3
"""Aggregate the block-local SDAR runs into tab:block_privacy_grid and the SDAR
columns of the merged block table (tab:block_privacy_unsplit).

  python aggregate_block_grid.py --logs DIR [--logs DIR ...] --out-dir results

Reads only the run logs (no grids): final SSIM and, for the DP rows, the victim's
fg_loss trace. Cells are the mean SSIM over runs, UNGATED, reported as % of the
matched Raw SSIM (0.572 at the 4096-d cut / level 7, 0.832 at the 8192-d cut /
level 4), per the table caption. A DP run whose victim collapsed
(final fg_loss > 0.20) is dropped loudly.

Logs are matched by their parsed configuration: block_random runs fill b in
{1,2,4}; fixed_random runs at the same CR x lambda grid fill b=8 (the global
projection); runs with dp_sigma fill the +DP rows.

Outputs:
  tab_block_privacy_grid.csv   cut, cr, block, w0 ... w0.1  (percent of Raw)
  tab_block_privacy_sdar.csv   block, cr, dp_sigma, l7_pct, l4_pct  (merged-table SDAR columns)
"""
import argparse, csv, glob, os, re
from collections import defaultdict

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
HDR = re.compile(r"Start (\w+)/\w+ experiments on (\w+) at l(\d+)")
COMP = re.compile(r"^Compression defense: (.*)$", re.M)
C_MODE = re.compile(r"mode=(\w+)")
C_CR = re.compile(r"compression_ratio=" + NUM)
C_WCC = re.compile(r"wcc_lambda=" + NUM)
C_BLK = re.compile(r"block_size=(\d+)")
DP = re.compile(r"DP defense: dp_clip_norm=" + NUM + r", dp_sigma=" + NUM)
SSIM = re.compile(r"Average SSIM over all client's images:\s*" + NUM)
FG = re.compile(r"fg_loss: " + NUM)
RAW = {7: 0.572, 4: 0.832}
WCCS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
FG_MAX = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cells = defaultdict(list)   # (level, cr, block, wcc, dp) -> [ssim]
    for d in args.logs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.log"), recursive=True)):
            t = open(p, errors="ignore").read()
            h, comp, s = HDR.search(t), COMP.search(t), SSIM.findall(t)
            if not h or not comp or not s or h.group(1) != "usl":
                continue
            c = comp.group(1)
            mm = C_MODE.search(c)
            if not mm or mm.group(1) not in ("block_random", "fixed_random"):
                continue
            blk = int(C_BLK.search(c).group(1)) if (mm.group(1) == "block_random" and C_BLK.search(c)) else 8
            cr = int(float(C_CR.search(c).group(1)))
            wcc = float(C_WCC.search(c).group(1)) if C_WCC.search(c) else 0.0
            dp = DP.search(t)
            sigma = float(dp.group(2)) if dp else 0.0
            if sigma:
                fg = FG.findall(t)
                if fg and float(fg[-1]) > FG_MAX:
                    print(f"  [drop victim-collapsed] {os.path.basename(p)} fg={float(fg[-1]):.3f}")
                    continue
            cells[(int(h.group(3)), cr, blk, wcc, sigma)].append(float(s[-1].rstrip(".")))

    def pct(level, cr, blk, wcc, sigma=0.0):
        v = cells.get((level, cr, blk, wcc, sigma))
        return "" if not v else f"{(sum(v) / len(v)) / RAW[level] * 100:.2f}"

    with open(os.path.join(args.out_dir, "tab_block_privacy_grid.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cut", "cr", "block"] + [f"w{x:g}" for x in WCCS])
        for level, cut in ((7, "4096-d"), (4, "8192-d")):
            for cr in (8, 16, 32):
                for blk in (1, 2, 4, 8):
                    w.writerow([cut, cr, blk] + [pct(level, cr, blk, x) for x in WCCS])

    with open(os.path.join(args.out_dir, "tab_block_privacy_sdar.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "cr", "dp_sigma", "l7_pct", "l4_pct"])
        for cr in (8, 16, 32):
            for blk in (1, 2, 4, 8):
                w.writerow([blk, cr, 0, pct(7, cr, blk, 0.0), pct(4, cr, blk, 0.0)])
        for blk in (1, 2, 4, 8):
            w.writerow([blk, 8, 0.02, pct(7, 8, blk, 0.0, 0.02), pct(4, 8, blk, 0.0, 0.02)])
    n = sum(len(v) for v in cells.values())
    print(f"[aggregate_block_grid] {len(cells)} cells / {n} runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
