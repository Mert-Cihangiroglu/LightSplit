# Upstream provenance — FSHA

Derived from the authors' reference implementation of the Feature-Space
Hijacking Attack:

- **Repository:** https://github.com/pasquini-dario/SplitNN_FSHA
- **Base commit:** `bd895c58ec5c72f6b4ba9d426ab11615f146e5b2`

Upstream's licence is preserved verbatim in `LICENSE`. We retain the public
FSHA loop, decoder, discriminator, and WGAN objective unchanged.

## Files we added

- `bottlenecks.py` — the defense bottlenecks (`fixed_random`, `block_random`,
  `topk_sparse`, `learned_1x1`) and the within-class-compaction loss.
- `run_fsha_bottleneck.py` — single-configuration runner.
- `scripts/` — the grid drivers (`run_bottleneck_grid.py`,
  `run_clean_accuracy_grid.py`, `run_dpblock.py`, `run_clean_dpblock.py`)
  and metric computation.
- `test_topk_sparse_tf.py` — unit check.

## Files we modified

- `FSHA.py` — insert the bottleneck at the cut, carry the auxiliary loss, and
  apply DP on the transmitted payload as a per-configuration $\ell_2$ clip
  calibrated to the median payload norm plus Gaussian noise.
- `architectures.py`, `datasets.py`, `FSHA_distcor.py` — the extra cut
  geometry and datasets used in our grids.
