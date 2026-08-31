#!/usr/bin/env python3
"""
bench_compute.py: Computation efficiency benchmark for LightSplit bottleneck methods.

Compares topk_sparse, fixed_random (flat), and learned_1x1 at matched
communication budgets (same transmitted float count k, or honest byte-matched
for topk_sparse which must also send indices).

Default setup mirrors the ResNet-18 split at layer2 exit:
  activation_dim = 4096  (64 ch * 8 * 8 spatial)
  spatial_shape  = (64, 8, 8)
  batch_size     = 64

FLOPs definition: multiply-accumulate operations (MACs) per sample.
  Backward FLOPs include grad_input only for fixed_random (R is frozen, no
  weight grad), and grad_input + grad_weight for learned_1x1's compress conv.
  topk_sparse backward is a straight-through mask multiply (d MACs).

Usage:
    conda run -n LightSplit python the utility harness/bench_compute.py
    conda run -n LightSplit python the utility harness/bench_compute.py --ks 512 256 128 64
    python the utility harness/bench_compute.py --device cpu
"""
import argparse
import math
import sys
import time
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent

import torch
import torch.nn as nn
from typing import Dict, Tuple

# ── constants matching train_split.py ─────────────────────────────────────────
VALUE_BYTES = 4   # float32
INDEX_BYTES = 2   # uint16 index per kept coordinate

# ── bottleneck modules (self-contained) ───────────────────────────────────────

class TopKSparseBottleneck(nn.Module):
    def __init__(self, k_s: int):
        super().__init__()
        self.k_s = k_s

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        k = min(self.k_s, h.shape[1])
        _, idx = torch.topk(h.abs(), k, dim=1)
        mask = torch.zeros_like(h).scatter(1, idx, 1.0)
        return h * mask, {}


class FixedRandomBottleneck(nn.Module):
    def __init__(self, activation_dim: int, k: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(activation_dim, k))
        self.register_buffer("R", q)
        self.k = k

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        return h @ self.R, {}


class Learned1x1Bottleneck(nn.Module):
    def __init__(self, in_channels: int, k_channels: int):
        super().__init__()
        self.compress   = nn.Conv2d(in_channels, k_channels, 1)
        self.decompress = nn.Conv2d(k_channels, in_channels, 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        return self.compress(z), {}


# ── theoretical MAC calculators (per sample) ──────────────────────────────────
#
# Forward:
#   topk_sparse : abs(d) + partial-sort O(d) + scatter(k_s) + mask-mul(d)  ~= 3d
#   fixed_random: h @ R  ->  d * k  MACs
#   learned_1x1 : Conv2d(C->k_ch, 1x1) on (C,H,W)  ->  k_ch * C * H * W  MACs
#
# Backward (compress only; server-side lift_back not included):
#   topk_sparse : straight-through mask multiply  ->  d  MACs
#   fixed_random: grad_h = grad_z @ R^T  ->  k * d  MACs  (R frozen, no weight grad)
#   learned_1x1 : grad_input  ->  k_ch * C * H * W  MACs
#                 grad_weight ->  C * k_ch * H * W   MACs
#                 total bwd   ->  2 * k_ch * C * H * W  MACs

def mac_topk_fwd(d: int, k_s: int) -> int:
    # abs(d) + partial-sort O(d log k_s) + scatter(k_s) + mask-mul(d)
    return d + int(d * math.log2(max(k_s, 2))) + k_s + d

def mac_topk_bwd(d: int, k_s: int) -> int:
    return d                 # straight-through mask multiply

def mac_fr_fwd(d: int, k: int) -> int:
    return d * k

def mac_fr_bwd(d: int, k: int) -> int:
    return k * d             # R is frozen; only grad_input via R^T matmul

def mac_l1x1_fwd(C: int, k_ch: int, H: int, W: int) -> int:
    return k_ch * C * H * W

def mac_l1x1_bwd(C: int, k_ch: int, H: int, W: int) -> int:
    return 2 * k_ch * C * H * W   # grad_input + grad_weight


def human(n: int, precision: int = 2) -> str:
    """Human-readable MACs with suffix."""
    if n >= 1_000_000_000: return f"{n/1e9:.{precision}f}G"
    if n >= 1_000_000:     return f"{n/1e6:.{precision}f}M"
    if n >= 1_000:         return f"{n/1e3:.{precision}f}K"
    return str(n)


def human_latex(n: int) -> str:
    """LaTeX-friendly SI notation."""
    if n >= 1_000_000_000: return rf"{n/1e9:.1f}\,G"
    if n >= 1_000_000:     return rf"{n/1e6:.2f}\,M"
    if n >= 1_000:         return rf"{n/1e3:.1f}\,K"
    return str(n)


# ── timing harness ────────────────────────────────────────────────────────────

def time_fn(fn, warmup: int, iters: int, device: str) -> float:
    """Returns mean latency in milliseconds."""
    if device == "cuda":
        start_e = torch.cuda.Event(enable_timing=True)
        end_e   = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start_e.record()
        for _ in range(iters):
            fn()
        end_e.record()
        torch.cuda.synchronize()
        return start_e.elapsed_time(end_e) / iters
    else:
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000 / iters


# ── per-k benchmark ───────────────────────────────────────────────────────────

def run_one_k(k: int, bsz: int, d: int, C: int, H: int, W: int,
              warmup: int, iters: int, device: str) -> dict:

    # k_s for topk_sparse: byte-matched to k float32s (each kept coord = 4+2 bytes)
    k_s  = max(1, (k * VALUE_BYTES) // (VALUE_BYTES + INDEX_BYTES))
    # k_ch for learned_1x1: mirrors train_split.py
    k_ch = max(1, C // (d // k)) if k < d else C

    # honest transmitted float-equivalents
    tx_topk = math.ceil(k_s * (VALUE_BYTES + INDEX_BYTES) / VALUE_BYTES)
    tx_fr   = k
    tx_l1x1 = k_ch * H * W

    # ── topk_sparse ──────────────────────────────────────────────────────────
    topk_net  = TopKSparseBottleneck(k_s=k_s).to(device)
    h_flat    = torch.randn(bsz, d, device=device)
    h_nd      = h_flat.detach()

    def _topk_fwd():
        with torch.no_grad():
            topk_net(h_nd)

    def _topk_train():
        hh = h_flat.requires_grad_(True)
        topk_net(hh)[0].sum().backward()

    t_topk_fwd   = time_fn(_topk_fwd,   warmup, iters, device)
    t_topk_train = time_fn(_topk_train, warmup, iters, device)

    # ── fixed_random ─────────────────────────────────────────────────────────
    fr_net = FixedRandomBottleneck(activation_dim=d, k=k).to(device)

    def _fr_fwd():
        with torch.no_grad():
            fr_net(h_nd)

    def _fr_train():
        hh = h_flat.requires_grad_(True)
        fr_net(hh)[0].sum().backward()

    t_fr_fwd   = time_fn(_fr_fwd,   warmup, iters, device)
    t_fr_train = time_fn(_fr_train, warmup, iters, device)

    # ── learned_1x1 ──────────────────────────────────────────────────────────
    l1x1_net  = Learned1x1Bottleneck(in_channels=C, k_channels=k_ch).to(device)
    z_spatial = torch.randn(bsz, C, H, W, device=device)
    z_nd      = z_spatial.detach()

    def _l1x1_fwd():
        with torch.no_grad():
            l1x1_net(z_nd)

    def _l1x1_train():
        zz = z_spatial.requires_grad_(True)
        l1x1_net(zz)[0].sum().backward()

    t_l1x1_fwd   = time_fn(_l1x1_fwd,   warmup, iters, device)
    t_l1x1_train = time_fn(_l1x1_train, warmup, iters, device)

    # ── theoretical MACs (per sample) ────────────────────────────────────────
    mf_topk  = mac_topk_fwd(d, k_s)
    mb_topk  = mac_topk_bwd(d, k_s)
    mf_fr    = mac_fr_fwd(d, k)
    mb_fr    = mac_fr_bwd(d, k)
    mf_l1x1  = mac_l1x1_fwd(C, k_ch, H, W)
    mb_l1x1  = mac_l1x1_bwd(C, k_ch, H, W)

    return dict(
        k=k, k_s=k_s, k_ch=k_ch,
        tx_topk=tx_topk, tx_fr=tx_fr, tx_l1x1=tx_l1x1,
        # inference latency (ms, full batch)
        t_topk_fwd=t_topk_fwd, t_fr_fwd=t_fr_fwd, t_l1x1_fwd=t_l1x1_fwd,
        # training latency (fwd+bwd, ms, full batch)
        t_topk_train=t_topk_train, t_fr_train=t_fr_train, t_l1x1_train=t_l1x1_train,
        # MACs per sample: forward / backward / total
        mf_topk=mf_topk, mb_topk=mb_topk, mt_topk=mf_topk+mb_topk,
        mf_fr=mf_fr,     mb_fr=mb_fr,     mt_fr=mf_fr+mb_fr,
        mf_l1x1=mf_l1x1, mb_l1x1=mb_l1x1, mt_l1x1=mf_l1x1+mb_l1x1,
    )


# ── pretty-print helpers ──────────────────────────────────────────────────────

def print_table(title, header, rows, sep_width=96):
    SEP = "-" * sep_width
    print(f"\n{title}")
    print(SEP)
    print(header)
    print(SEP)
    for row in rows:
        print(row)


# ── LaTeX table builder ───────────────────────────────────────────────────────

def latex_table(results: list, bsz: int, device_name: str) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    cap = (
        r"\caption{Training-time bottleneck efficiency at matched communication budget "
        r"($k$ transmitted floats per sample, batch 64, ResNet-18 split at layer~2 "
        rf"exit, {device_name})."
        r" MACs are per sample (forward + backward through the compress stage only;"
        r" server-side lift not included)."
        r" \textsc{TopK-Sparse} keeps the $k_s$ largest-magnitude coordinates plus "
        r"2-byte uint16 indices; \textsc{Fixed-Rand} projects with a frozen orthonormal"
        r" matrix ($\mathbf{R}$ is never updated); \textsc{Learned-1$\times$1} applies"
        r" a trainable 1$\times$1 convolution.}"
    )
    lines.append(cap)
    lines.append(r"\label{tab:bottleneck_compute}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & $k$ & Tx floats & MACs\textsubscript{fwd} "
        r"& MACs\textsubscript{train} & Latency (ms) \\"
    )
    lines.append(r"\midrule")

    method_names = [
        r"\textsc{TopK-Sparse}",
        r"\textsc{Fixed-Rand}",
        r"\textsc{Learned-1$\times$1}",
    ]

    for i, r in enumerate(results):
        k = r["k"]
        rows_data = [
            # (method_name, tx, mf, mt, t_train)
            (method_names[0], r["tx_topk"], r["mf_topk"], r["mt_topk"], r["t_topk_train"]),
            (method_names[1], r["tx_fr"],   r["mf_fr"],   r["mt_fr"],   r["t_fr_train"]),
            (method_names[2], r["tx_l1x1"], r["mf_l1x1"], r["mt_l1x1"], r["t_l1x1_train"]),
        ]
        for j, (mname, tx, mf, mt, tt) in enumerate(rows_data):
            kstr = str(k) if j == 0 else ""
            lines.append(
                rf"{mname} & {kstr} & {tx} "
                rf"& {human_latex(mf)} & {human_latex(mt)} & {tt:.2f} \\"
            )
        if i < len(results) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bottleneck compute benchmark")
    parser.add_argument("--ks",     type=int, nargs="+", default=[512, 256, 128, 64],
                        help="Transmitted float budgets to sweep")
    parser.add_argument("--bsz",    type=int, default=64,   help="Batch size")
    parser.add_argument("--d",      type=int, default=4096, help="Flat activation dim (C*H*W)")
    parser.add_argument("--C",      type=int, default=64,   help="Spatial channels")
    parser.add_argument("--H",      type=int, default=8,    help="Spatial height")
    parser.add_argument("--W",      type=int, default=8,    help="Spatial width")
    parser.add_argument("--warmup", type=int, default=50,   help="Warmup iterations")
    parser.add_argument("--iters",  type=int, default=200,  help="Timed iterations")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dev_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"\nBench: device={args.device} ({dev_name})  bsz={args.bsz}  "
          f"d={args.d}  spatial=({args.C},{args.H},{args.W})  "
          f"warmup={args.warmup}  iters={args.iters}\n")

    results = []
    for k in args.ks:
        print(f"  k={k} ...", end="", flush=True)
        r = run_one_k(k, bsz=args.bsz, d=args.d, C=args.C, H=args.H, W=args.W,
                      warmup=args.warmup, iters=args.iters, device=args.device)
        results.append(r)
        print(f"  topk=({r['t_topk_fwd']:.2f}/{r['t_topk_train']:.2f})ms  "
              f"fr=({r['t_fr_fwd']:.2f}/{r['t_fr_train']:.2f})ms  "
              f"l1x1=({r['t_l1x1_fwd']:.2f}/{r['t_l1x1_train']:.2f})ms  "
              f"[fwd/train]")

    W96 = 96

    # ── Table 1: MACs per sample (fwd and training) ───────────────────────────
    hdr = (f"{'k':>5}  {'method':>16}  {'Tx floats':>9}  "
           f"{'MACs fwd':>12}  {'MACs bwd':>12}  {'MACs train':>12}")
    rows = []
    for r in results:
        k = r["k"]
        rows.append(f"{k:>5}  {'topk_sparse':>16}  {r['tx_topk']:>9}  "
                    f"{human(r['mf_topk']):>12}  {human(r['mb_topk']):>12}  {human(r['mt_topk']):>12}")
        rows.append(f"{'':>5}  {'fixed_random':>16}  {r['tx_fr']:>9}  "
                    f"{human(r['mf_fr']):>12}  {human(r['mb_fr']):>12}  {human(r['mt_fr']):>12}")
        rows.append(f"{'':>5}  {'learned_1x1':>16}  {r['tx_l1x1']:>9}  "
                    f"{human(r['mf_l1x1']):>12}  {human(r['mb_l1x1']):>12}  {human(r['mt_l1x1']):>12}")
        rows.append("")
    print_table("TABLE 1: Theoretical MACs per sample", hdr, rows, W96)

    # ── Table 2: Training latency (ms, full batch) ───────────────────────────
    hdr = (f"{'k':>5}  {'topk_sparse':>14}  {'fixed_random':>14}  {'learned_1x1':>14}")
    rows = []
    for r in results:
        rows.append(
            f"{r['k']:>5}  {r['t_topk_train']:>14.3f}  "
            f"{r['t_fr_train']:>14.3f}  {r['t_l1x1_train']:>14.3f}"
        )
    print_table("TABLE 2: Training latency (ms, full batch of 64)  [fwd + bwd, compress only]",
                hdr, rows, W96)

    # ── Table 3: Training throughput (samples/sec) ────────────────────────────
    hdr = f"{'k':>5}  {'topk_sparse':>14}  {'fixed_random':>14}  {'learned_1x1':>14}"
    rows = []
    for r in results:
        b = args.bsz
        rows.append(
            f"{r['k']:>5}  {b/(r['t_topk_train']/1000):>14,.0f}  "
            f"{b/(r['t_fr_train']/1000):>14,.0f}  "
            f"{b/(r['t_l1x1_train']/1000):>14,.0f}"
        )
    print_table("TABLE 3: Training throughput (samples/sec)", hdr, rows, W96)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print(f"\n{'=' * W96}")
    print("LATEX TABLE (ready to paste):")
    print("=" * W96)
    print(latex_table(results, bsz=args.bsz, device_name=dev_name))
    print("=" * W96)

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks_plot     = [r["k"] for r in results]
    mac_topk    = [r["mt_topk"]  / 1e3 for r in results]   # in K
    mac_fr      = [r["mt_fr"]    / 1e3 for r in results]
    mac_l1x1    = [r["mt_l1x1"]  / 1e3 for r in results]
    lat_topk    = [r["t_topk_train"] for r in results]
    lat_fr      = [r["t_fr_train"]   for r in results]
    lat_l1x1    = [r["t_l1x1_train"] for r in results]

    COLORS = {"topk": "#d62728", "fr": "#1f77b4", "l1x1": "#2ca02c"}
    MARKERS = {"topk": "^", "fr": "s", "l1x1": "o"}
    LABELS  = {
        "topk": "TopK-Sparse",
        "fr":   "Fixed-Rand",
        "l1x1": r"Learned $1{\times}1$",
    }

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
    })

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    ax2 = ax.twinx()

    # primary axis: training MACs (log), solid lines
    for key, mac_data in [("topk", mac_topk), ("fr", mac_fr), ("l1x1", mac_l1x1)]:
        ax.plot(ks_plot, mac_data,
                color=COLORS[key], marker=MARKERS[key],
                linestyle="-", label=LABELS[key])
    ax.set_yscale("log")
    ax.set_xlabel("$k$ (transmitted floats)")
    ax.set_ylabel("Training MACs / sample (K)")
    ax.set_xticks(ks_plot)
    ax.yaxis.label.set_color("black")

    # secondary axis: training latency (linear), dashed lines, same colors
    for key, lat_data in [("topk", lat_topk), ("fr", lat_fr), ("l1x1", lat_l1x1)]:
        ax2.plot(ks_plot, lat_data,
                 color=COLORS[key], marker=MARKERS[key],
                 linestyle="--", alpha=0.55, linewidth=1.1)
    ax2.set_ylabel("Training latency (ms)", color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555", labelsize=7)
    ax2.yaxis.label.set_color("#555555")

    # legend: one entry per method from primary axis + style hint
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    hint_solid = Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.2, label="MACs (left)")
    hint_dash  = Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2, label="Latency (right)")
    ax.legend(handles=handles + [hint_solid, hint_dash],
              labels=labels + ["MACs (left)", "Latency (right)"],
              frameon=False, fontsize=7, ncol=1,
              loc="upper left")

    fig.tight_layout()
    out_path = HERE / "figures" / "compute_efficiency.pdf"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    print(textwrap.dedent("""
    FLOP definitions:
      topk_sparse fwd : abs(d) + partial-sort O(d log k_s) + scatter(k_s) + mask-mul(d)
      topk_sparse bwd : straight-through mask multiply  = d  MACs
      fixed_random fwd: h @ R  (d x k matmul)  = d*k  MACs
      fixed_random bwd: grad_h = grad_z @ R^T  = k*d  MACs  (R frozen, no weight update)
      learned_1x1 fwd : Conv2d(C->k_ch, 1x1)  = k_ch*C*H*W  MACs
      learned_1x1 bwd : grad_input + grad_weight  = 2*k_ch*C*H*W  MACs
      Training = fwd + bwd (compress only; server-side lift_back excluded).
    """).strip())


if __name__ == "__main__":
    main()
