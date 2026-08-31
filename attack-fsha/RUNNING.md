# attack-fsha — FSHA fully-malicious feature hijacking

FSHA is the active-attack stress test: a *malicious* server hijacks the feature
space during training rather than passively observing it. Upstream attribution
is in `UPSTREAM.md`; upstream's own README is preserved as `README.md`.

TensorFlow. Use the `lightsplit-tf` environment.

## Entry point

`run_fsha_bottleneck.py` runs a single configuration:

```bash
python run_fsha_bottleneck.py \
  --iterations 10000 --batch-size 64 --examples 20 \
  --compression-ratio 8 --wcc-lambda 0.1 --seed 123
```

Run `python run_fsha_bottleneck.py --help` for the bottleneck-mode flag and the
learning rates (`--lr-f`, `--lr-tilde`, `--lr-D`).

The paper's operating point is the level-4 ResNet cut
($4{\times}4{\times}256 = 4096$-d, matching the CIFAR cut dimension) at
CR $8\times$, i.e. $k = 512$.

## Grids

`scripts/` holds the drivers:

- `run_bottleneck_grid.py` — the defense grid (attack runs)
- `run_clean_accuracy_grid.py` — matching clean-accuracy runs
- `run_dpblock.py`, `run_clean_dpblock.py` — the DP and block-random cells
- `compute_reconstruction_metrics.py` — MSE / SSIM / PSNR / LPIPS

DP here is a per-configuration $\ell_2$ clip calibrated to the **median payload
norm**, plus Gaussian noise — implemented inline in `FSHA.py`, not as a separate
bottleneck class.

## Reproducing the paper's tables

`reproduce/tab_fsha_active.sh` and `reproduce/tab_block_privacy_fsha.sh` train
every cell of the two FSHA tables and write one CSV each; `../REPRODUCE.md`
gives the commands, their cost and the column meanings.
