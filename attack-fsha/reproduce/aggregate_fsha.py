#!/usr/bin/env python3
"""Turn the FSHA reproduction runs into the paper's table cells.

  --table fsha_active         -> dataset, method, acc, mse, ssim, lpips, delta_raw (mean and std over seeds)
  --table block_privacy_fsha  -> dataset, block, cr, dp_sigma, ssim (mean and std over seeds)

Scoring is scripts/compute_reconstruction_metrics.py applied to each run's reconstructions.npz:
images mapped from [-1, 1] to [0, 1]; MSE, SSIM (11x11 Gaussian window), LPIPS (AlexNet, on [-1, 1]);
mean over the 20 images of a run. Cells are the mean and population std over seeds. delta_raw is
LPIPS / LPIPS(Raw, same seed), averaged over seeds. Accuracy is best clean test accuracy of the
single clean run. Needs torch and lpips.

--runs-map JSON overrides the run directories: {"attack": {dataset: {method: {seed: dir}}},
"clean": {dataset: {method: dir}}}; entries not given fall back to the standard layout.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import compute_reconstruction_metrics as M  # noqa: E402

ROWS = [  # (method, attack run-name suffix, relative case dir)
    ("Raw",            "",     "raw/no_defense"),
    ("Raw+DP",         "_dp",  "raw/no_defense_dp"),
    ("Learned 1x1",    "",     "cr08/learned_1x1/wcc0"),
    ("top-K sparse",   "",     "cr08/topk_sparse/wcc0"),
    ("LS-F",           "",     "cr08/fixed_random/wcc0"),
    ("LS-F+WCC(0.1)",  "",     "cr08/fixed_random/wcc0p1"),
    ("LS-F+WCC(0.3)",  "",     "cr08/fixed_random/wcc0p3"),
    ("LS-F+DP",        "_dp",  "cr08/fixed_random/wcc0_dp"),
]

def score(run_dir):
    z = np.load(Path(run_dir) / "reconstructions.npz")
    ou, ru = M.to_unit(z["original"]), M.to_unit(z["recovered"])
    return (float(np.mean(M.mse_per_image(ou, ru))), float(np.mean(M.ssim_per_image(ou, ru))),
            float(np.mean(M.lpips_per_image(ou * 2 - 1, ru * 2 - 1))))

def acc(run_dir):
    p = Path(run_dir) / "metrics.json"
    return 100.0 * json.load(open(p))["best_clean_test_accuracy"] if p.exists() else float("nan")

def ms(v):
    v = np.asarray(v, float); return float(v.mean()), float(v.std())

def fsha_active(root, seeds, datasets, rmap):
    rows = []
    for ds in datasets:
        per = {}
        for method, suf, rel in ROWS:
            per[method] = {}
            for s in seeds:
                d = rmap.get("attack", {}).get(ds, {}).get(method, {}).get(str(s)) \
                    or root / "tab_fsha_active" / f"{ds}_seed{s}{suf}" / rel
                if Path(d, "reconstructions.npz").exists():
                    per[method][s] = score(d)
                else:
                    print(f"[missing] {d}", file=sys.stderr)
        for method, suf, rel in ROWS:
            got = sorted(per[method])
            cd = rmap.get("clean", {}).get(ds, {}).get(method) \
                or root / "tab_fsha_active_clean" / f"{ds}_seed123{suf}" / rel
            row = {"dataset": ds, "method": method, "acc": acc(cd), "n_seeds": len(got), "seeds": " ".join(map(str, got))}
            if got:
                v = np.array([per[method][s] for s in got])
                dr = [per[method][s][2] / per["Raw"][s][2] for s in got if s in per["Raw"]]
                for i, k in enumerate(("mse", "ssim", "lpips")):
                    row[k], row[k + "_std"] = ms(v[:, i])
                row["delta_raw"], row["delta_raw_std"] = ms(dr) if dr else (float("nan"), float("nan"))
            rows.append(row)
    return rows, ["dataset", "method", "acc", "mse", "mse_std", "ssim", "ssim_std", "lpips", "lpips_std",
                  "delta_raw", "delta_raw_std", "n_seeds", "seeds"]

def block_privacy_fsha(root, seeds, datasets, rmap):
    """block_random b in {1,2,4} x CR in {8,16,32} and the b in {1,2,4} + DP cells at CR 8: SSIM of the
    20 reconstructed images per run, mean and population std over seeds."""
    rows = []
    for ds in datasets:
        cells = [(b, cr, "0", f"cr{cr:02d}/block_random/b{b}", "_block") for b in (1, 2, 4) for cr in (8, 16, 32)]
        cells += [(b, 8, "0.02", f"cr08/block_random/b{b}_dp", "_blockdp") for b in (1, 2, 4)]
        for b, cr, sig, rel, suf in cells:
            vals = []
            for sd in seeds:
                d = rmap.get("attack", {}).get(ds, {}).get(f"b{b}_cr{cr}_dp{sig}", {}).get(str(sd)) \
                    or root / "tab_block_privacy_fsha" / f"{ds}_seed{sd}{suf}" / rel
                if Path(d, "reconstructions.npz").exists():
                    vals.append(score(d)[1])
                else:
                    print(f"[missing] {d}", file=sys.stderr)
            m, sdv = ms(vals) if vals else (float("nan"), float("nan"))
            rows.append({"dataset": ds, "block": b, "cr": cr, "dp_sigma": sig, "ssim": m, "ssim_std": sdv, "n_seeds": len(vals)})
    return rows, ["dataset", "block", "cr", "dp_sigma", "ssim", "ssim_std", "n_seeds"]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=["fsha_active", "block_privacy_fsha"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist"])
    ap.add_argument("--runs-map", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rmap = json.load(open(a.runs_map)) if a.runs_map else {}
    rows, fields = (fsha_active(a.root, a.seeds, a.datasets, rmap) if a.table == "fsha_active"
                    else block_privacy_fsha(a.root, a.seeds, a.datasets, rmap))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
