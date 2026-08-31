# attack-sdar — SDAR passive reconstruction

SDAR is the primary privacy attack in the paper: a passive server trains a
substitute encoder on auxiliary public data plus a decoder against the captured
cut-layer payload. Upstream attribution is in `UPSTREAM.md`.

TensorFlow. Use the `lightsplit-tf` environment from the top-level README.

## Entry point

`scripts/run_sdar.py`, with five positional arguments. It resolves `src/` through
a relative path, so **run it from inside `scripts/`**:

```
run_sdar.py  <setting>  <model>  <dataset>  <level>  <run>
             vsl|usl    resnet   cifar10    4..9     run number
             plainnet   cifar100
```

The paper's SDAR experiments use CIFAR-10 and CIFAR-100. (The upstream loader
also recognises other dataset names, which the paper does not use.)

The paper uses level 7 ($L{=}4$, $d{=}4096$) and level 4 ($L{=}7$, $d{=}8192$).
USL is the headline setting (labels hidden from the server); VSL is the
label-aware sanity check.

```bash
cd scripts
python run_sdar.py usl resnet cifar10 7 1 \
  --bottleneck_mode fixed_random --compression_ratio 8
```

## Defense flags

| Flag | Effect |
| --- | --- |
| `--bottleneck_mode` | `none`, `fixed_random` (**LightSplit-F**), `block_random`, `topk_sparse`, `learned_1x1` |
| `--compression_ratio R` / `--bottleneck_k K` | payload size |
| `--block_size B` | tile size for `block_random` |
| `--wcc_lambda L` | within-class compaction |
| `--center_loss_lambda L`, `--center_alpha A` | center loss |
| `--dp_clip_norm C`, `--dp_sigma S` | DP on the payload |
| `--width`, `--attacker_width`, `--server_width` | cut/attacker/server width |

**LightSplit-L** (`fixed_random_ll`) is not here — see
`attack-sdar-lightsplit-l/`. Narrowed cut widths beyond the `--width` presets
are in `attack-sdar-cutwidth/`.

## Sweeps and reporting

The `scripts/run_*.sh` drivers reproduce the paper's grids. `aggregate_*.py`
and `compute_blackbox_*_metrics.py` turn run output into the metric CSVs, and
`make_wcc_cr_grid.py` aggregates the WCC sweep.

Each run writes its reconstruction figure and arrays under `scripts/<mode>_results/`
(upstream's layout); the summary lines the aggregators read are on stdout.
Reconstruction `.npy` / `.png` dumps are excluded — regenerate by re-running.

## Reproducing the paper's tables

`reproduce/tab_sdar_all_split_points.sh` re-runs the merged USL+VSL sweep (and its
derived compact, DP and consolidated-table cells); `reproduce/tab_block_privacy_grid.sh`
the block-local grid and the SDAR columns of the merged block table. See
`../REPRODUCE.md` for commands, cost, protocols, and verification.
