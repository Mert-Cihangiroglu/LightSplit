# utility — main split-learning harness

Every accuracy number in the paper comes from here, plus the WCC, DP,
block-local, cut-width and top-$k$ sweeps, the ablations, and backdoor detection.

## Entry point

`train_split.py`. Per-dataset architectures and cut geometry live in
`client_models.py` under `MODEL_CONFIGS`.

```bash
python train_split.py \
  --dataset cifar10 --mode fixed_random --compression-ratio 16 \
  --num-clients 10 --alpha 1.0 --epochs 200 --patience 30 \
  --deep-head --gpu 0 --no-viz \
  --summary-csv results/cifar10_lsf_cr16.csv
```

Datasets: `cifar10`, `cifar100`, `fmnist`, `gtsrb`.

## Modes

`--mode` accepts `raw`, `fixed_random` (**LightSplit-F**), `fixed_random_ll`
(**LightSplit-L**), `block_random`, `block_random_ll`, `topk_sparse`,
`learned_1x1`, `fixed_random_resid` and `fixed_random_conv`.

It defaults to `fixed_random` (**LightSplit-F**).

## Key flags

| Flag | Effect |
| --- | --- |
| `--compression-ratio N` | `k = d // N` |
| `--block-size B` | tile size for `block_random*` |
| `--wcc-lambda L` | within-class compaction strength |
| `--center-lambda L`, `--center-alpha A` | center loss, for the WCC comparison |
| `--clip-norm C --noise-multiplier S` | DP: per-sample $\ell_2$ clip + Gaussian noise |
| `--cut-channels C` | narrowed cut layer (cut-width axis) |
| `--mlp-hidden H` | lift-back MLP width for `*_ll` modes |
| `--deep-head` / `--client-heads` | deeper head (L2H) / per-client heads (PCH) |
| `--num-clients N --alpha A` | client count and Dirichlet non-IID concentration |

`--sweep` with `--modes`, `--compression-ratio-list`, `--num-clients-list` or
`--alpha-list` runs a grid in one process.

## Sweeps

The `run_*.sh` drivers reproduce the paper's grids — `run_wcc_*.sh`,
`run_dp_*.sh`, `run_util_block_random*.sh`, `run_cutwidth_grid.sh`,
`run_matched_byte_baseline.sh`, `run_ablation_*.sh`. Each writes a
`summary.csv` under `results/`.

These legacy drivers carry a `conda activate` line and paths from the setup
they were written on; prefer the `reproduce/` drivers below, or read the
`python ...` invocation inside and run `train_split.py` directly with the flags
above.

## Backdoor detection

`run_backdoor.py` with `backdoor_{data,split,observer,metrics}.py`; driver
`run_backdoor_sweep.sh`.

## Reproducing the paper's tables

`reproduce/tab_full_sweep.sh` re-runs the full utility sweep
(`tab:full_sweep_baseline_appendix` and the utility columns of the compact
comparison table) with the paper's per-cell epoch budgets, and
`reproduce/fig_cutwidth_utility.sh` the utility panel of the cut-width figure.
`reproduce/tab_block_util_FL.sh` and `reproduce/tab_dp_composition.sh` rebuild
the block-local utility grid and the DP-composition table;
`reproduce/tab_wcc_ablation.sh`, `reproduce/tab_mlp_hidden.sh` and
`reproduce/tab_client_scaling.sh` the three ablation tables;
`reproduce/tab_backdoor.sh` the backdoor-detection table and
`reproduce/tab_comm_cost.py` the communication-cost table.
See `../REPRODUCE.md` for the exact commands, cost, and output columns.
