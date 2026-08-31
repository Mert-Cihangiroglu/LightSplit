"""
Spatially-resolved SDAR reconstruction analysis (offline, no re-training).

Motivation: a single global mean SSIM conflates *how much* of an image the attack
recovered with *how well* the recovered part matches. A recon that is partly
near-perfect and partly flat grey can score the same mean SSIM as a uniformly
mediocre one, yet they leak very differently. We decompose the mean into:

  coverage      = fraction of pixels whose local (windowed) SSIM >= TAU
                  ("how much of the image actually leaked")
  cond_ssim     = mean local SSIM over that recovered support
                  ("how exact are the parts that came back") -- only meaningful
                  paired with coverage; alone it is cherry-picking.
  cond_psnr     = PSNR over the recovered support (dB)
  best_patch    = max local SSIM anywhere (worst-case leak)
  global_ssim   = plain mean of the local map (sanity vs the logged scalar)

Support definition (locked with user): local SSIM >= 0.6.

Recon pixels live only in the saved matplotlib grids (1 row of 20 tiles,
bbox_inches='tight' -> thin black spine + small white gutters, which we trim).
Originals are not saved for usl, but load_single_batch is fully deterministic
(random_state=42, shuffle seed=420, .take(1)) so x[10:30] reproduces the exact
20 paired originals at native 32x32.

Run from the scripts/ dir with the sdar env:
  python analyze_recon_locality.py
"""
import os
import sys
import glob
import csv
import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

sys.path.append('../src')
import data  # noqa: E402  (needs ../src on path, matches run_sdar.py)

TAU = 0.6            # local-SSIM threshold defining the recovered support
SIGMA = 1.5          # gaussian window, matches tf.image.ssim(filter_sigma=1.5)
C1 = 0.01 ** 2
C2 = 0.03 ** 2
TRIM = 0.09          # fraction cropped from each tile edge (spine + gutter)
FIGROOT = "defense_results"
RAWROOT = "sdar_results"


def load_originals(dataset="cifar10", batch_size=128, t=10, n=20):
    ds = data.load_single_batch(dataset, batch_size=batch_size)
    x = [xb for xb, _ in iter(ds)][0].numpy().astype(np.float32)  # (B,32,32,3) in [0,1]
    return x[t:t + n]


def split_recon(path, n=20):
    """Split a 1-row recon grid into n tiles, trim spine/gutter, resize to 32x32."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    H, W, _ = img.shape
    tiles = []
    for i in range(n):
        x0 = round(i * W / n)
        x1 = round((i + 1) * W / n)
        t = img[:, x0:x1, :]
        h, w, _ = t.shape
        cy, cx = int(h * TRIM), int(w * TRIM)
        t = t[cy:h - cy, cx:w - cx, :]
        t = Image.fromarray((t * 255).astype(np.uint8)).resize((32, 32), Image.BILINEAR)
        tiles.append(np.asarray(t).astype(np.float32) / 255.0)
    return np.stack(tiles, axis=0)


def ssim_map(a, b):
    """Per-channel gaussian-windowed SSIM map, averaged over channels -> (H,W)."""
    maps = []
    for c in range(a.shape[-1]):
        x, y = a[..., c], b[..., c]
        mux, muy = gaussian_filter(x, SIGMA), gaussian_filter(y, SIGMA)
        mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
        sx = gaussian_filter(x * x, SIGMA) - mux2
        sy = gaussian_filter(y * y, SIGMA) - muy2
        sxy = gaussian_filter(x * y, SIGMA) - muxy
        m = ((2 * muxy + C1) * (2 * sxy + C2)) / ((mux2 + muy2 + C1) * (sx + sy + C2))
        maps.append(m)
    return np.mean(maps, axis=0)


def is_paired(originals, rec, tol=0.8, abs_gap=0.10):
    """The eval-batch shuffle order varies per run; we regenerated exactly one
    ordering. A recon PNG is usable only if its tiles correspond index-for-index
    to our originals. Compare identity-order mean SSIM to the Hungarian optimal
    assignment. A *structured* recon that is misaligned shows a large gap (the
    optimal permutation recovers real matches the identity order misses ~0.13+);
    an aligned recon OR a failed/mush recon shows a small gap (identity is already
    optimal, or nothing matches anything so both are low). So we keep a run when
    identity is essentially optimal, by either a relative OR an absolute-gap test;
    a large gap means real structure sitting under a different ordering we cannot
    reconstruct -> exclude. For mush recons coverage is pairing-insensitive, so
    the identity pairing we then use is safe. Returns (paired, identity, best)."""
    n = len(originals)
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = ssim_map(originals[i], rec[j]).mean()
    identity = float(np.mean(np.diag(M)))
    ri, ci = linear_sum_assignment(-M)
    best = float(np.mean(M[ri, ci]))
    paired = best <= 1e-9 or identity >= tol * best or (best - identity) <= abs_gap
    return paired, identity, best


def cell_metrics(originals, recon_paths):
    rows, n_paired, excluded = [], 0, []
    for p in recon_paths:
        rec = split_recon(p)
        paired, ident, best = is_paired(originals, rec)
        if not paired:
            excluded.append((os.path.basename(p), ident, best))
            continue
        n_paired += 1
        for i in range(len(originals)):
            S = ssim_map(originals[i], rec[i])
            supp = S >= TAU
            cov = float(supp.mean())
            err = ((originals[i] - rec[i]) ** 2).mean(axis=-1)  # (H,W) mse over channels
            if supp.any():
                cond_ssim = float(S[supp].mean())
                cmse = float(err[supp].mean())
                cond_psnr = -10.0 * math.log10(cmse) if cmse > 0 else float("nan")
            else:
                cond_ssim = cond_psnr = float("nan")
            rows.append((float(S.mean()), cov, cond_ssim, cond_psnr, float(S.max())))
    a = np.array(rows, dtype=float) if rows else np.zeros((0, 5))
    return {
        "n_paired": n_paired,
        "n_total": len(recon_paths),
        "n_img": len(a),
        "excluded": excluded,
        "global_ssim": np.nanmean(a[:, 0]) if len(a) else float("nan"),
        "coverage": np.nanmean(a[:, 1]) if len(a) else float("nan"),
        "cond_ssim": np.nanmean(a[:, 2]) if len(a) else float("nan"),
        "cond_psnr": np.nanmean(a[:, 3]) if len(a) else float("nan"),
        "best_patch": np.nanmean(a[:, 4]) if len(a) else float("nan"),
    }


def g(pattern):
    return sorted(glob.glob(pattern))


def get_cells(group):
    if group == "wcc":
        fr = f"{FIGROOT}/fixed_random/usl/cifar10/fig"
        return "recon_locality_cr8_l7.csv", [
            ("Undefended (raw L7)", g(f"{RAWROOT}/usl/cifar10/fig/resnet_l7_run*.png")),
            ("Projection only (cr8, lambda=0)", g(f"{fr}/cr8_resnet_l7_run*.png")),
            ("WCC lambda=0.01", g(f"{fr}/cr8_wcc0.01_resnet_l7_run*.png")),
            ("WCC lambda=0.1", g(f"{fr}/cr8_wcc0.1_resnet_l7_run*.png")),
            ("Center lambda=0.01", g(f"{fr}/cr8_center0.01_resnet_l7_run*.png")),
            ("Center lambda=0.1", g(f"{fr}/cr8_center0.1_resnet_l7_run*.png")),
        ]
    if group == "topk_sparse":
        ts = f"{FIGROOT}/topk_sparse/vsl/cifar10/fig"
        return "recon_locality_topk_sparse_l7.csv", [
            ("Undefended (raw L7, vsl)", g(f"{RAWROOT}/vsl/cifar10/fig/resnet_l7_run*.png")),
            ("topk_sparse CR=8", g(f"{ts}/cr8_resnet_l7_run*.png")),
            ("topk_sparse CR=16", g(f"{ts}/cr16_resnet_l7_run*.png")),
            ("topk_sparse CR=32", g(f"{ts}/cr32_resnet_l7_run*.png")),
        ]
    raise SystemExit(f"unknown group {group!r}")


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "topk_sparse"
    originals = load_originals()
    out, cells = get_cells(group)

    out_rows, all_excluded = [], []
    hdr = f"{'defense':<34}{'paired':>7}{'globalSSIM':>12}{'coverage':>10}{'condSSIM':>10}{'condPSNR':>10}{'bestPatch':>11}"
    print(f"\nSDAR reconstruction locality  [{group}]  (CIFAR-10, L7; support = local SSIM >= {TAU})")
    print(hdr)
    print("-" * len(hdr))
    for name, paths in cells:
        if not paths:
            print(f"{name:<34}{'--':>7}   (no recon PNGs found)")
            continue
        m = cell_metrics(originals, paths)
        pr = f"{m['n_paired']}/{m['n_total']}"
        all_excluded += [(name,) + e for e in m["excluded"]]
        if m["n_paired"] == 0:
            print(f"{name:<34}{pr:>7}   (no verified-paired runs)")
            continue
        print(f"{name:<34}{pr:>7}{m['global_ssim']:>12.3f}{m['coverage']*100:>9.1f}%"
              f"{m['cond_ssim']:>10.3f}{m['cond_psnr']:>10.2f}{m['best_patch']:>11.3f}")
        out_rows.append({"defense": name, "paired": pr, "n_img": m["n_img"],
                         "global_ssim": round(m["global_ssim"], 4),
                         "coverage": round(m["coverage"], 4),
                         "cond_ssim": round(m["cond_ssim"], 4),
                         "cond_psnr": round(m["cond_psnr"], 3),
                         "best_patch": round(m["best_patch"], 4)})
    if all_excluded:
        print("\nExcluded (eval-batch ordering not recoverable from our single regenerated batch):")
        for name, fn, ident, best in all_excluded:
            print(f"  {name:<32} {fn:<40} identity={ident:.3f} best-assign={best:.3f}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["defense", "paired", "n_img", "global_ssim",
                                          "coverage", "cond_ssim", "cond_psnr", "best_patch"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {os.path.abspath(out)}")
    print("Note: coverage = fraction of image with local SSIM>=%.2f; cond_* are computed ONLY on\n"
          "that support (cherry-picked by construction) so they must be read together with coverage." % TAU)


if __name__ == "__main__":
    main()
