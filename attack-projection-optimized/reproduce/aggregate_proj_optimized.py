#!/usr/bin/env python3
"""Build the projection-optimized attacker table from the two gradfusion summary CSVs.

Rows are the fused signal sets, columns the compression ratios; every cell is
reconstruction SSIM on CIFAR-10 (higher = stronger attack).

  z        experiment 3, R denied           (unknown_ssim)
  z+zRT    experiment 3, R known            (known_ssim)
  z+g      experiment 6, fuse_pinv=False    (img_ssim)
  z+g+zRT  experiment 6, fuse_pinv=True     (img_ssim)
"""
import argparse, csv

CRS = (8, 16, 32)


def load(path):
    return list(csv.DictReader(open(path)))


def cell(rows, exp, cr, col):
    for r in rows:
        if r["experiment"] == exp and int(r["cr"]) == cr and r.get(col) not in (None, "", "nan"):
            return float(r[col])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="gradfusion summary.csv (experiments 3 and 6)")
    ap.add_argument("--summary-pinv", required=True, help="gradfusion_pinv summary.csv (experiment 6 --fuse-pinv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, rows_pinv = load(a.summary), load(a.summary_pinv)
    table = [
        ("z",       lambda cr: cell(rows, "exp3_known_vs_unknown", cr, "unknown_ssim")),
        ("z+zRT",   lambda cr: cell(rows, "exp3_known_vs_unknown", cr, "known_ssim")),
        ("z+g",     lambda cr: cell(rows, "exp6_grad_fusion", cr, "img_ssim")),
        ("z+g+zRT", lambda cr: cell(rows_pinv, "exp6_grad_fusion", cr, "img_ssim")),
    ]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signals"] + [f"cr{c}" for c in CRS])
        for name, fn in table:
            w.writerow([name] + [f"{fn(cr):.5f}" for cr in CRS])
    print(f"[aggregate_proj_optimized] 4 rows -> {a.out}")


if __name__ == "__main__":
    main()
