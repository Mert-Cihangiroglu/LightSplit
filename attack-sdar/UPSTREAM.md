# Upstream provenance — SDAR

Derived from the authors' reference implementation of SDAR:

- **Repository:** https://github.com/zhxchd/SDAR_SplitNN
- **Base commit:** `3f0a0a8e119c9c12113b538bf3c120bd183de771`

Upstream's licence is preserved verbatim in `LICENSE`. The SDAR attack loop,
simulator/discriminator pair, decoder, and training schedule are upstream's; we
did not alter the attack itself.

## Files we added

- `src/sdar/bottlenecks.py` — all defense bottlenecks (`fixed_random`,
  `block_random`, `topk_sparse`, `learned_1x1`), the within-class-compaction
  loss, the center loss, and the DP clip-and-noise operator.
- `scripts/` — the sweep drivers (`run_*.sh`), metric computation
  (`compute_blackbox_*.py`) and aggregation (`aggregate_*.py`,
  `make_wcc_cr_grid.py`).

## Files we modified

- `scripts/run_sdar.py` — CLI flags for the defense: `--bottleneck_mode`,
  `--compression_ratio`, `--bottleneck_k`, `--block_size`, `--wcc_lambda`,
  `--center_loss_lambda`, `--center_alpha`, `--dp_clip_norm`, `--dp_sigma`.
- `src/sdar/sdar.py`, `src/sdar/sdar_hetero.py` — insert the bottleneck at the
  cut and add the auxiliary losses to the victim's objective.
- `src/models/resnet.py` — additional cut-width options.
