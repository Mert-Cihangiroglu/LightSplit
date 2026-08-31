# Reproducing the paper's tables and figures

One section per exhibit. Each gives the environment, the exact command, its cost,
and the CSV it produces; the CSV's columns are the exhibit's cells. Run
`python verify_tables.py expected/<exhibit>.json <csv>` to compare a reproduction
with the printed numbers (mean and std over seeds are compared at the paper's
precision; GAN-based attacks vary run to run, so expect agreement in ranking and
at the printed precision for the multi-seed means, not bit-exact values).

Environments: `tf` = Python with TensorFlow 2.9 (SDAR, FSHA); `torch` = Python
with PyTorch 2.x, `lpips`, `piq` (everything else, and all reconstruction
scoring). Set `CUDA_VISIBLE_DEVICES` to pick the GPU.

## FSHA (`attack-fsha/`)

### Table `tab:fsha_active` — fully-malicious FSHA at CR 8x

```
cd attack-fsha
FSHA_TF_PYTHON=<tf python> FSHA_TORCH_PYTHON=<torch python> bash reproduce/tab_fsha_active.sh
```
Trains the 8 rows x 2 datasets x 3 seeds (48 attack runs, 10 000 iterations each) and
the 16 clean-accuracy runs, then writes `results/reproduce/tab_fsha_active.csv`.
About 4.5 GPU-hours. Columns: `acc`, `mse`, `ssim`, `lpips`, `delta_raw` (means over
seeds, each with a `_std` column); `delta_raw` is LPIPS / LPIPS(Raw) at the same seed.

### FSHA columns of `tab:block_privacy_unsplit` — block-local projection under FSHA

```
cd attack-fsha
FSHA_TF_PYTHON=<tf python> FSHA_TORCH_PYTHON=<torch python> bash reproduce/tab_block_privacy_fsha.sh
```
b in {1,2,4} x CR in {8,16,32} and b in {1,2,4} with DP (sigma 0.02) at CR 8, seeds 123/124/125,
2 datasets; writes `results/reproduce/tab_block_privacy_fsha.csv` (`ssim` mean over seeds,
`ssim_std`, `n_seeds`). About 6.5 GPU-hours.

Both scripts skip cells whose output already exists, so they can be resumed.
`ITERATIONS`, `CLEAN_EPOCHS`, `SEEDS`, `DATASETS` and `OUT` can be overridden for a
smoke run, e.g. `ITERATIONS=20 CLEAN_EPOCHS=1 SEEDS=123 DATASETS=mnist`.

## ImageNet-1k (`utility-imagenet/`)

Both scripts need `IMAGENET_ROOT` (ImageNet-1k with `train/` and `val/` ImageFolder
subdirectories; not included) and a `torch` Python in `PYTHON`. `--resume` makes every
arm skip if its summary row exists; `ARMS="raw block_b2_cr8"` selects arms, so the
independent runs can be spread over GPUs. `FAKE=1 EPOCHS=1` is a pipeline smoke test.

### Figure `fig:imagenet_generality` — top-1 as % of Raw at CR 8/16/32

```
cd utility-imagenet
IMAGENET_ROOT=/data/imagenet PYTHON=<torch python> bash reproduce/fig_imagenet_generality.sh
```
13 runs (Raw; block b=2, top-k sparse, learned 1x1, global R at CR 8, 16, 32), 90 epochs
each, about one GPU-day per run on an RTX A6000. Writes `results/reproduce/fig_imagenet_generality.csv`
with `best_top1` (best validation top-1 over the run) and `pct_of_raw`, the plotted value.

### Figure `fig:imagenet_lsl_lsf` — LightSplit-L against LightSplit-F at CR 8

```
cd utility-imagenet
IMAGENET_ROOT=/data/imagenet PYTHON=<torch python> bash reproduce/fig_imagenet_lsl_lsf.sh
```
Four matched LS-L / LS-F pairs (Adam 30 and 100 epochs; SGD 90 epochs; SGD 577 epochs
on 200 images per class); the SGD-90 LS-F arm is taken from the previous figure when
present. Writes `results/reproduce/fig_imagenet_lsl_lsf.csv`: per pair the two best
top-1 values, their ratio (`ratio_best_pct`, the numbers quoted in the text), the mean
per-epoch ratio over the last tenth of the run (`ratio_final_tenth_pct`) and the ratio
of final top-1 (`ratio_final_pct`; the block b=2 reference line in panel (b) uses this).
The paper's Adam-100 pair stopped at epoch 89 and the 200-per-class pair at epochs 542/544;
the script runs the full schedules.

## GPT-2 (`utility-llm/`)

### Figure `fig:llm_generality` — perplexity inflation over Raw, GPT-2 124M to 1.5B

```
cd utility-llm
PYTHON=<python with torch, transformers, datasets> bash reproduce/fig_llm_generality.sh
```
28 runs: for each of gpt2, gpt2-medium, gpt2-large, gpt2-xl, a Raw run and LightSplit-F /
LightSplit-L at CR 8, 16, 32, each one epoch of ~20.5M WikiText-103 tokens with the cut
after round(0.1 x n_layer) blocks. GPT-2 weights and WikiText-103 come from the Hugging Face
hub (`HF_HOME` selects the cache). About one GPU-hour for gpt2 and ~10 for gpt2-xl.
Writes `results/reproduce/fig_llm_generality.csv`: `best_val_ppl` and `inflation`
(= best validation perplexity / the same model's Raw), the plotted value. Runs are
independent; `MODELS="gpt2 gpt2-medium"` selects a subset, `SMOKE=1` is a pipeline test.

## UCI HAR (`utility-har/`)

### Figures `fig:har_cr8`, `fig:har_generality`, `fig:har_privacy` — tabular utility and privacy

```
cd utility-har
PYTHON=<torch python> bash reproduce/fig_har.sh
```
57 cells: Raw and six methods (LightSplit-F, +WCC 0.1, LightSplit-L, +WCC 0.1, learned 1x1,
top-K sparse) at CR 8/16/32 (k = 32/16/8 on the 256-d cut), 3 seeds, 120 epochs each,
followed by the three attacks of `attacks.py`. Small MLPs: about 40 s per cell on CPU.
UCI HAR is downloaded on first use (`HAR_DATA` sets the cache directory). Writes
`results/reproduce/fig_har.csv`: per method and CR the mean over seeds of `test_acc`
(best-checkpoint test accuracy, %), `recon_r2` (median per-feature R^2 of the decoder probe)
and `reid` (subject re-identification accuracy), each with `_pstd` (population std, used
for the accuracy bars) and `_std` (sample std, used for the two privacy bars). The
main-text figure is the CR 8 slice.

## Cut width (`attack-sdar-cutwidth/`, `utility/`)

### Figure `fig:cutwidth` — privacy (a) and utility (b) across the cut width d

Panel (a), SDAR reconstruction:
```
cd attack-sdar-cutwidth
SDAR_TF_PYTHON=<tf python> bash reproduce/fig_cutwidth_privacy.sh
```
Level-7 CIFAR-10 USL attack with the client head's cut stage at C in {8,16,32,64,128}
channels (d = 512 ... 8192); per width a Raw run and LightSplit-F at CR 8/16/32, 5 seeds:
100 runs of 20 000 iterations, about 1.3 GPU-hours each. Writes
`results/reproduce/fig_cutwidth_privacy.csv`; the plotted value is `ssim_max`, the
strongest attack over the victim-healthy seeds (`n_victim_ok`); `ssim_median` and the
attacker-convergence flags are alongside. Each SDAR run also drops its reconstruction
figure and arrays under `scripts/<mode>_results/`, upstream's layout. `SDAR_ITERS=200 SEEDS=0 WIDTHS=cut8` is a smoke run.

Panel (b), accuracy:
```
cd utility
PYTHON=<torch python> bash reproduce/fig_cutwidth_utility.sh
```
Same widths, `--head-fixed-body` (native-width head body, only the cut stage varies),
single client, 100 epochs, seeds 42/43/123: 60 runs, about 15 GPU-minutes each. Writes
`results/reproduce/fig_cutwidth_utility.csv` (`test_acc` = mean best held-out accuracy).

## UnSplit (`attack-unsplit/`)

### Table `tab:unsplit_attack_full`, UnSplit columns of `tab:block_privacy_unsplit`, `tab:privacy_all`, `tab:main-compact`

```
cd attack-unsplit
PYTHON=<torch python> bash reproduce/tab_unsplit.sh
```
One script covers every UnSplit cell: per (dataset, cut) Raw, Learned 1x1, LightSplit-F x
lambda_WCC in {0, 0.01, 0.1} and top-k sparse at CR 8/16/32; the DP rows (sigma 0.02, 10 images,
seed 0: Raw+DP at both cuts, LightSplit-F+DP at CR 8, at cut 4 as the global b = 8 tile); and at
cut 2 the block-local projection b in {1,2,4} x CR and b in {1,2,4}+DP at CR 8. 88 single-image
cells of a few minutes each and 10 ten-image cells; about 16 GPU-hours in total, `JOBS` cells
at a time (default 2). MNIST and Fashion-MNIST are downloaded into the run directory unless
`DATA_ROOT` points at an existing torchvision folder.

Writes `results/reproduce/tab_unsplit.csv`, one row per cell: `acc`, whole-image `mse`, `ssim`,
`psnr`, `lpips`, foreground `mse_fg`, and `mse_ratio` / `mse_fg_ratio` against the Raw cell of the
same (dataset, cut) (`mse_fg_pct` is the percentage printed in the main tables). The `*0` columns
hold the metrics of image 0 alone; the paper's ten-image DP rows print those for the whole-image
metrics and, at cut 4, for the MSE_fg ratio. DP clip values are the ones the paper's runs used
(median payload norm of the untrained victim; `reproduce/measure_dp_norms.py` is the calibration).

The paper's single-image cells were run unseeded, so individual cells vary from run to run
(a Raw or Learned 1x1 SSIM can move by 0.05 to 0.15 between identical runs) while the
LightSplit-F plateau (identical reconstruction at every CR and lambda) and the ten-image DP
rows reproduce at the printed precision. `expected/tab_unsplit_full.json` and
`expected/tab_block_privacy_unsplit.json` hold the printed values. `SMOKE=1` runs a
one-epoch, three-iteration subset; `RSEEDS="0 1 2 3"` adds extra R seeds at (cut 4, CR 8,
lambda 0.1).

## FORA (`attack-fora/`)

### Table `tab:fora-full`, FORA columns of `tab:block_privacy_unsplit`, `tab:privacy_all`, `tab:main-compact`

```
cd attack-fora
PYTHON=<torch python> GPUS="0 1 2 3" bash reproduce/tab_fora.sh
```
FORA against the CIFAR-10 MobileNet cut at level 2 (d = 16 384), 40 000 attack iterations per
run: Raw, Raw+DP, Learned 1x1 with 8/4/2 channels, LightSplit-F at k in {2048, 1024, 512} x
lambda_WCC in {0, 0.001 (k 2048), 0.01, 0.1}, top-k sparse, LightSplit-F+DP (sigma 0.02, clip =
median payload norm) and the block-local projection b in {1, 2, 4} x k. Each cell runs the
attacker seeds the paper averaged (listed per cell in `reproduce/aggregate_fora.py`: three for
most cells, one for the lambda_WCC = 0.01 rows, the 16x/32x lambda_WCC = 0.1 rows and the block
rows); 61 runs of about 8 GPU-hours each, one run per GPU in `GPUS`. CIFAR-10 is downloaded into
the run directory unless `DATA_ROOT` points at an existing torchvision folder.

Writes `results/reproduce/tab_fora.csv`, one row per cell, each run scored on the 32-image
reconstruction grid it saves at its last evaluation: `pseudo_*` for FORA's surrogate-trained
decoder and `ref_*` for the white-box reference decoder (`ssim`, `psnr`, `lpips`), means over
the cell's runs; the pseudo columns average the converged runs only (a run counts as collapsed
when its pseudo SSIM is below -0.3 and its PSNR below 9 dB while another run of the cell
reconstructs; `n_runs` / `n_pseudo` give the counts). `ref_lpips_ratio` is the full table's
Delta_Raw; `pseudo_lpips_pct` and `pseudo_ssim_pct` are the FORA columns of the main tables;
`victim_acc` is the victim's last test accuracy from the log. `SMOKE=1` runs three cells for
100 iterations.

## UCI HAR under label heterogeneity (`utility-har/`)

### Table `tab:har_noniid`

```
cd utility-har
PYTHON=<torch python> bash reproduce/tab_har_noniid.sh
```
Five clients with Dirichlet label skew, alpha in {0.5, 0.3, 0.1}: Raw, top-k sparse, Learned 1x1,
LightSplit-F and LightSplit-L each with lambda_WCC in {0, 0.1}, at k in {32, 16, 8}, seeds 0 1 2,
200 epochs; 171 CPU runs of a few minutes. Writes `results/reproduce/tab_har_noniid.csv`: the Raw
row carries accuracies, every other row the difference in points to Raw at the same alpha
(`a0.5_delta`, `a0.3_delta`, `a0.1_delta`, with `_n` seed counts). At alpha 0.1 seed 2 is left
out of every arm (`EXCLUDE=0.1:2`), as in the paper. The alpha -> infinity column (`iid_delta`)
is read from the single-client sweep of `reproduce/fig_har.sh` (`IID_CSV`, default
`results/reproduce/fig_har.csv`).

## Full utility sweep (`utility/`)

### Table `tab:full_sweep_baseline_appendix`, utility columns of `tab:main-compact`

```
cd utility
PYTHON=<torch python> JOBS=2 bash reproduce/tab_full_sweep.sh
```
U-shaped SL with 10 clients: Raw, LightSplit-F, LightSplit-L, Learned 1x1 and top-k sparse on
CIFAR-10, CIFAR-100, FashionMNIST and GTSRB, non-IID (alpha 0.1; 0.5 for CIFAR-100) and IID,
all four head configurations (L1H/L2H x SCH/PCH), CR in {8, 16, 32}: 416 runs. Epoch budgets
match the paper's runs cell by cell (the script's `budget` function): 2000 epochs with early
stopping (patience 200) for FashionMNIST/GTSRB, LightSplit-L and the CIFAR-100 LightSplit-F
PCH cells; 200 epochs for the remaining CIFAR-100 SCH cells; 100 epochs for CIFAR-10, the
CIFAR-100 PCH Raw/Learned cells and every top-k sparse cell. Expect several GPU-days at
JOBS=2; `DATASETS`/`MODES`/`CRS` restrict the grid for partial runs. GTSRB needs the folder
layout described in `utility/data.py`; the other datasets download automatically.

Writes `results/reproduce/tab_full_sweep.csv` (one row per method x CR x depth x head: best
test accuracy per dataset x distribution, and its difference to the matched Raw cell) and
`results/reproduce/tab_full_sweep_main_utility.csv` (the L2H-SCH cells as percent change
relative to Raw: IID per dataset plus the mean over the four non-IID splits, i.e. the utility
columns of `tab:main-compact`).

```
python verify_tables.py expected/tab_full_sweep.json utility/results/reproduce/tab_full_sweep.csv
python verify_tables.py expected/tab_main_utility.json utility/results/reproduce/tab_full_sweep_main_utility.csv
```
In `expected/tab_full_sweep.json` five top-k GTSRB delta cells are omitted: the paper prints
them as differences of the rounded accuracies, one hundredth away from the full-precision
difference the aggregator emits; the accuracies themselves are checked.

## SDAR (`attack-sdar/`, `attack-sdar-lightsplit-l/`)

SDAR runs are TensorFlow (the `lightsplit-tf` environment); the aggregators need a
PyTorch environment with `lpips` (pass it as `AGG_PYTHON`). SDAR does not seed its runs:
each run id is an independent attacker/victim trajectory, so re-run cells scatter around
the printed values and `verify_tables.py` is a consistency check of the aggregation
pipeline rather than a bit-exact test. The aggregation protocols themselves (which run
feeds which cell, convergence gating, averaging) are fixed in the aggregators and were
validated against the paper's own run set.

### Tables `tab:final-sdar-usl-all-split-points`, `tab:sdar-main-compact`, `tab:dp_privacy`, SDAR pair of `tab:privacy_all`

```
cd attack-sdar
PYTHON=<tf python> AGG_PYTHON=<torch python> GPUS="0 1" bash reproduce/tab_sdar_all_split_points.sh
```
USL and VSL at the 4096-d (level 7) and 8192-d (level 4) cuts: Raw, Learned 1x1 and
LightSplit-F cells are one attacker run each; top-k sparse cells average the runs whose
attacker converged (label accuracy >= 0.40); the DP rows (sigma 0.02, clip = the median
payload norm per configuration) average 3 runs (6 for level-7 LightSplit-F CR8 + DP),
dropping a run whose victim or simulator collapsed. 153 runs, each a full SDAR training
of several hours on one GPU. Writes `reproduce/results/tab_sdar_all_split_points.csv`
(SSIM/PSNR/LPIPS_20 per protocol), `tab_sdar_compact.csv` (the USL level-4 CR8 slice with
MSE and the LPIPS ratio to Raw), `tab_dp_sdar_privacy.csv` (the DP SSIM cells) and
`tab_recon_main_sdar.csv` (the consolidated table's SDAR pair, percent change vs Raw).

```
python verify_tables.py expected/tab_sdar_all_split_points.json attack-sdar/reproduce/results/tab_sdar_all_split_points.csv
python verify_tables.py expected/tab_sdar_compact.json attack-sdar/reproduce/results/tab_sdar_compact.csv
python verify_tables.py expected/tab_dp_sdar.json attack-sdar/reproduce/results/tab_dp_sdar_privacy.csv
python verify_tables.py expected/tab_recon_main_sdar.json attack-sdar/reproduce/results/tab_recon_main_sdar.csv
```
LPIPS_20 is computed between the run's saved 20-image reconstruction grid and the
originals grid, so those cells exist only after the corresponding runs have written
their grids.

### Table `tab:block_privacy_grid`, SDAR columns of `tab:block_privacy_unsplit`

```
cd attack-sdar
PYTHON=<tf python> AGG_PYTHON=<torch python> GPUS="0 1 2 3" bash reproduce/tab_block_privacy_grid.sh
```
USL at both cuts: block_random b in {1,2,4} and the global projection (b=8, LightSplit-F)
over CR {8,16,32} x lambda_WCC {0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, two runs per cell,
plus the CR8 +DP rows (sigma 0.02). 480 runs. Cells are mean SSIM as percent of the
matched Raw, ungated, per the table caption; a +DP run whose victim collapsed is dropped.
Writes `reproduce/results/tab_block_privacy_grid.csv` and `tab_block_privacy_sdar.csv`.
The printed b=4 +DP cell at the 8192-d cut is a two-run mean; the expected file leaves
that cell unchecked because a third archived run with a collapsed attacker sits outside
the printed pair.

```
python verify_tables.py expected/tab_block_privacy_grid.json attack-sdar/reproduce/results/tab_block_privacy_grid.csv
python verify_tables.py expected/tab_block_privacy_sdar.json attack-sdar/reproduce/results/tab_block_privacy_sdar.csv
```

### Table `tab:sdar_lsl` — SDAR against LightSplit-L

```
cd attack-sdar-lightsplit-l
PYTHON=<tf python> AGG_PYTHON=<torch python> bash reproduce/tab_sdar_lsl.sh
```
USL level 7, `fixed_random_ll`, CR {8,16,32}; rows are means over 3/6/6 runs
(SSIM and PSNR from the run logs, LPIPS_20 from the saved grids). The Raw and
LightSplit-F reference rows come from `attack-sdar`'s table above. 15 runs.

```
python verify_tables.py expected/tab_sdar_lsl.json attack-sdar-lightsplit-l/reproduce/results/tab_sdar_lsl.csv
```

## Projection-optimized attacks (`attack-projection-optimized/`)

### Table `tab:proj_opt` — passive attackers that know the projection

```
cd attack-projection-optimized
PYTHON=<torch python> bash reproduce/tab_proj_optimized.sh
```
CIFAR-10, seed 42. Pass 1 trains one split model per CR (2000 epochs, early
stop at patience 100) and runs experiment 3 (decoder on z, with and without R)
and experiment 6 (decoder on z plus the returned cut-layer gradient g); pass 2
reuses the same checkpoints for experiment 6 with `--fuse-pinv` (z + g + zR^T),
decoder-only. Several GPU-hours on one GPU. Columns are reconstruction SSIM at
CR 8/16/32 for the four fused signal sets. `SMOKE=1` runs one CR at toy budgets
to check mechanics.

```
python verify_tables.py expected/tab_proj_optimized.json attack-projection-optimized/reproduce/results/tab_proj_optimized.csv
```

## Block-local and DP utility grids (`utility/`)

### Table `tab:block_util_F`/`tab:block_util_L` — block-local utility over tile size

```
cd utility
PYTHON=<torch python> JOBS=4 bash reproduce/tab_block_util_FL.sh
```
528 runs (dataset x distribution x head config x CR x b {1,2,4} x {transpose,
learned} lift-back) at 2000 epochs / patience 100 — several GPU-weeks; shard
with `DATASETS=`/`HEADS=`/`BLOCKS=`. GTSRB skips b=4 (untileable). Requires the
full utility sweep's Raw rows (`tab_full_sweep.sh`) for the reference cells.
Cells are accuracy-point deltas vs the matched Raw; a cell covered by several
archived runs is their mean. Nine GTSRB cells whose printed deltas were
computed from rounded accuracies are left unchecked in the expected file.
`SMOKE=1` runs one cell at 2 epochs.

```
python verify_tables.py expected/tab_block_util_FL.json utility/results/reproduce/tab_block_util_FL.csv
```

### Table `tab:dp_utility`/`tab:dp_block` — utility cost of DP composition

```
cd utility
PYTHON=<torch python> JOBS=4 bash reproduce/tab_dp_composition.sh
```
Raw+DP, global LightSplit-F/L+DP and block-local b {1,2,4}+DP (clip 10,
sigma 0.02), 2000 epochs / patience 100 (Raw+DP CIFAR-10: 100 epochs, matching
its no-DP twin) — several GPU-weeks. Deltas read against the no-DP references,
so run `tab_full_sweep.sh` and `tab_block_util_FL.sh` first. One GTSRB cell
whose printed delta was computed from rounded accuracies is left unchecked.
`SMOKE=1` runs two cells at 2 epochs.

```
python verify_tables.py expected/tab_dp_composition.json utility/results/reproduce/tab_dp_composition.csv
```

## Utility ablations (`utility/`)

### Table `tab:mlp-hidden-ablation` — LightSplit-L hidden width

```
cd utility
PYTHON=<torch python> JOBS=4 bash reproduce/tab_mlp_hidden.sh
```
88 runs (4 datasets x 4 head configs x 4-5 widths + Raw baselines) at
2000 epochs / patience 200, CR 8, IID; a few GPU-days. Each row's Raw
reference is the grid's own raw run, falling back to the full sweep's cell.
`SMOKE=1` runs two cells at 2 epochs.

```
python verify_tables.py expected/tab_mlp_hidden.json utility/results/reproduce/tab_mlp_hidden.csv
```

### Table `tab:client_scaling_ablation` — client-count scaling

```
cd utility
PYTHON=<torch python> JOBS=4 bash reproduce/tab_client_scaling.sh
```
18 runs (CIFAR-10/100 x {Raw, LightSplit-F, LightSplit-L} x n {25,50,100})
at 2000 epochs / patience 200, CR 8, L2H-SCH, IID; a few GPU-days. The n=10
column is read from the full utility sweep (`tab_full_sweep.sh`).

```
python verify_tables.py expected/tab_client_scaling.json utility/results/reproduce/tab_client_scaling.csv
```

### Table `tab:wcc_ablation` — WCC across head configs, global vs b=2

```
cd utility
PYTHON=<torch python> JOBS=4 bash reproduce/tab_wcc_ablation.sh
```
68 runs (F/L x {global, b=2} x 4 head configs x lambda {0,1e-3,1e-2,1e-1}
plus 4 raw anchors) at 2000 epochs / patience 100, CIFAR-10, CR 8, IID; a few
GPU-days. Global rows read against the full sweep's Raw cells
(`tab_full_sweep.sh`); the b=2 rows read against this grid's own raw runs,
per the table caption.

```
python verify_tables.py expected/tab_wcc_ablation.json utility/results/reproduce/tab_wcc_ablation.csv
```

### Table `tab:bd-ours` — label-free backdoor detection

```
cd utility
PYTHON=<torch python> bash reproduce/tab_backdoor.sh
```
30 runs (baseline: 3 seeds at 50 epochs; each ablation row changes one
parameter, single runs at 30 epochs; fixed_random plus its fixed_random_ll
twin, CR 16); roughly a GPU-day. The detector applies per-epoch MAD z-scores
to each client's backbone-output cosine and flags mean |z| > 3; the baseline
cell is the mean F1 over seeds.

```
python verify_tables.py expected/tab_backdoor.json utility/results/reproduce/tab_backdoor.csv
```

## Analytic and derived tables

### Table `tab:comm_cost` — split-point communication cost

Pure arithmetic (no training):
```
python utility/reproduce/tab_comm_cost.py --out utility/results/reproduce/tab_comm_cost.csv
python verify_tables.py expected/tab_comm_cost.json utility/results/reproduce/tab_comm_cost.csv
```

### Tables derived from the reproduced grids

Several printed tables are arithmetic summaries of cells that the exhibits
above already reproduce; they need no separate runs:

- `tab:main-compact` (Table II) privacy columns: percent change vs the Raw cell
  of the same grid — SDAR from the USL 8192-d SSIM cells of
  `tab_sdar_all_split_points.csv`, FORA from `attack-fora`'s pseudo-decoder
  cells, UnSplit from the `mse_fg_ratio` split-2 cells of
  `tab_unsplit_full.csv`. The HAR column is the percent change of the
  `test_acc` cells of `utility-har`'s `fig_har.csv`.
- `tab:privacy_all` (Table III) UnSplit columns: `mse_fg_ratio - 1` of the
  CR 8 split-2 rows of `tab_unsplit_full.csv`; its tabular-probe columns are
  the percent change of `recon_r2` / `reid` in `fig_har.csv` (DP rows: the
  HAR DP exhibit below).
- `tab:unsplit_attack_headline`: accuracy range and worst foreground-MSE ratio
  over CR of the lambda=0 split-2 rows of `tab_unsplit_full.csv`.
- `tab:model_partitions` and `tab:wcc_wire_summary` describe the architecture
  (parameter counts of the model definitions shipped in each folder, and the
  cut-interface message layout); they contain no experimental numbers.

### HAR DP probe cells of `tab:privacy_all`

```
cd utility-har
PYTHON=<torch python> bash reproduce/tab_har_dp.sh
```
Six CPU runs (Raw+DP and LightSplit-F+DP, sigma 0.02, per-arm payload clips,
3 seeds), minutes each; the cell is the median over seeds of the probe R^2
and re-identification accuracy, as percent change vs the Raw row of
`fig_har.csv` (run `fig_har.sh` first). HAR retraining scatters these percent
cells by about a point run-to-run, so treat the comparison as a consistency
check rather than a bit-exact test.

```
python verify_tables.py expected/tab_har_dp.json utility-har/results/reproduce/tab_har_dp.csv
```
