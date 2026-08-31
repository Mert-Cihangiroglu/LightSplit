"""Median transmitted-payload L2 norm for the HAR cut, so the DP clip is calibrated the
same way as the FORA / UnSplit / SDAR arms: clip AT the median (~50% clip fraction),
noise_std = sigma * clip. Measured on the member (fit) set of the trained no-DP run."""
import sys, numpy as np, torch
from pathlib import Path
from attacks import load_run, split_views
for run in sys.argv[1:]:
    model, meta, cfg = load_run(Path(run))
    X = split_views(meta)["in"][0]
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(next(model.parameters()).device)).cpu()
    n = z.norm(dim=1)
    med = float(n.median())
    print(f"{run:>52}  liftback={cfg['liftback']:<7} payload_dim={z.shape[1]:>4}  "
          f"p10={np.percentile(n,10):7.3f}  median={med:7.3f}  p90={np.percentile(n,90):7.3f}  "
          f"-> clip={round(med,3)}")
