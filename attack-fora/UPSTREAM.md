# Upstream provenance — FORA

Derived from the authors' reference implementation of FORA:

- **Repository:** https://github.com/X1aoyangXu/FORA
- **Last upstream commit:** `92830f2` ("Update README.md")

The FORA attack — feature-oriented reconstruction, its surrogate client model
and decoder — is upstream's and unmodified in substance.

## Files we added

- `defenses.py` — `DefenseWrapper` and the defense bottlenecks
  (`fixed_random`, `block_random`, `topk_sparse`, `learned_1x1`), plus the
  within-class-compaction loss.
- `reproduce/` — the run script and aggregator for the paper's FORA tables.
- `measure_dp_norms.py` — payload-norm measurement used to calibrate the DP clip.
- `test_*.py` — unit checks for each defense.

## Files we modified

- `main.py` — wire the defense into the training loop and add its CLI flags.
- `model.py`, `utils.py` — insert the bottleneck at the cut; carry the
  auxiliary loss.
