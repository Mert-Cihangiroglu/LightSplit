# LightSplit — artifact

Code accompanying *LightSplit: Practical Privacy-Preserving Split Learning via
Orthogonal Projections*.

The main utility harness lives in `utility/`, with three further `utility-*/`
folders for the experiments that scale beyond CIFAR-scale image classification:
ImageNet-1k, tabular data, and language modelling. Every reconstruction attack we
evaluate against gets its own self-contained folder, `attack-*/`.

Each folder re-implements the defense natively inside its own codebase and
framework, so nothing imports across folders and each can be set up and run
independently.

---

## Naming: paper ↔ code

The paper names the two deployed mechanisms **LightSplit-F** and
**LightSplit-L**. The code predates those names and calls them `fixed_random`
and `fixed_random_ll`. The mapping is exact — same mechanism, different label:

| Paper | Code (`--mode` / `--bottleneck_mode`) |
| --- | --- |
| **LightSplit-F** | `fixed_random` |
| **LightSplit-L** | `fixed_random_ll` |
| LightSplit-F, block-local variant (tile size *b*) | `block_random` (+ `--block-size b`) |
| LightSplit-L, block-local variant | `block_random_ll` |
| Raw / no compression | `raw` (PyTorch) · `none` (TensorFlow) |
| top-$k$ sparsification (matched-byte baseline) | `topk_sparse` |
| Learned $1{\times}1$ convolution baseline | `learned_1x1` |
| WCC — within-class compaction, strength $\lambda_{\mathrm{WCC}}$ | `--wcc-lambda` / `--wcc_lambda` |
| Center loss (WCC comparison) | `--center-lambda` / `--center_loss_lambda` |
| DP — per-sample $\ell_2$ clip + Gaussian noise | `--clip-norm` + `--noise-multiplier` (PyTorch) · `--dp_clip_norm` + `--dp_sigma` (TensorFlow) |

`topk_sparse` is the only top-$k$ mechanism in this artifact; the token `topk`
in variable names and `torch.topk` calls refers to it. A few exploratory
switches in the code (`fixed_random_resid`, `fixed_random_conv`,
`--recon-lambda`, and similar) belong to mechanisms the paper does not report;
their defaults leave them off.

The compression ratio is `CR = d / k`, where `d` is the flattened cut-layer
dimension. Passing `--compression-ratio 16` sets `k = d // 16`.

---

## Layout

| Folder | What it is | Paper |
| --- | --- | --- |
| `utility/` | Main split-learning training harness. All accuracy numbers, WCC / DP / block / cut-width / top-$k$ sweeps, ablations, backdoor detection. | Main results table, full-sweep appendix, ablations, DP + block utility grids, backdoor table, communication cost |
| `utility-imagenet/` | ImageNet-1k scaling: ResNet split, 10 IID clients, 90 epochs, block-local projection with tile size $b$ as the utility dial. | ImageNet-1k subsection + figure |
| `utility-har/` | Tabular generality (HAR) and the tabular reconstruction / re-identification probes. The folder also carries loaders for the Purchase100 / Texas100 MIA benchmarks, which the paper does not report. | Tabular generality + tabular reconstruction privacy |
| `utility-llm/` | Language-modelling generality: GPT-2 split, validation perplexity vs compression. | Language-modelling paragraph |
| `attack-sdar/` | SDAR passive reconstruction (USL + VSL). The primary privacy attack. Covers the Raw baseline (`none`) and `fixed_random`, `block_random`, `topk_sparse`, `learned_1x1`, WCC, center loss, DP — all as flags on one codebase. | SDAR tables (compact + USL/VSL all-split-points), DP privacy, block privacy grid, top-$k$ matched-byte privacy |
| `attack-sdar-lightsplit-l/` | SDAR fork adding the `fixed_random_ll` bottleneck. | LightSplit-L privacy |
| `attack-sdar-cutwidth/` | SDAR fork with a narrowed cut layer (modified `src/models/resnet.py`). | Cut-width subsection + figure |
| `attack-fora/` | FORA feature-oriented reconstruction attack. | FORA main + appendix tables, block privacy |
| `attack-unsplit/` | UnSplit model-inversion and label-inference attack. | UnSplit tables (masked + whole-image), block privacy |
| `attack-fsha/` | FSHA fully-malicious feature-hijacking stress test. | FSHA active-attack table, block privacy under FSHA |
| `attack-projection-optimized/` | Attackers built around the projection: known-$R$ white-box inversion, gradient fusion, null-space rank analysis. | Projection-optimized attacker table + figure |

### Why three SDAR folders

`attack-sdar/` is the main one and covers every SDAR row in the paper. The
other two are small forks of it: `-lightsplit-l` adds the `fixed_random_ll`
bottleneck and touches `sdar.py` / `run_sdar.py`; `-cutwidth` adds the narrow
cut widths and touches `sdar.py` / `run_sdar.py` / `resnet.py`. Both edit the
same lines of those two files. They are kept separate so that each folder is
exactly the code that produced its numbers, rather than a merge made after the
fact.

---

## Environments

Two stacks, split by attack framework.

**PyTorch** — `utility/`, `attack-fora/`, `attack-unsplit/`,
`attack-projection-optimized/`:

```bash
conda create -n lightsplit python=3.10
conda activate lightsplit
pip install torch torchvision numpy pandas matplotlib
pip install piq tensorboard        # attack-fora only
```

Developed against torch 2.5.1.

**TensorFlow** — `attack-sdar*/`, `attack-fsha/`:

```bash
conda create -n lightsplit-tf python=3.10
conda activate lightsplit-tf
pip install tensorflow numpy pandas matplotlib scikit-image
```

Note: some run scripts carry a `conda activate` line naming the environment
from our internal setup. Change it to whichever environment you created, or
run the underlying `python ...` command directly.

---

## Quickstart

Train a LightSplit-F model at 16× compression:

```bash
cd utility
python train_split.py \
  --dataset cifar10 --mode fixed_random --compression-ratio 16 \
  --num-clients 10 --epochs 200 --patience 30 --deep-head --gpu 0 \
  --summary-csv results/cifar10_lsf_cr16.csv
```

LightSplit-L is the same command with `--mode fixed_random_ll`. Add
`--wcc-lambda 0.1` for WCC, or `--clip-norm 1.0 --noise-multiplier 0.02` for DP.

Run SDAR against a defended cut:

```bash
cd attack-sdar/scripts        # run_sdar.py resolves src/ via a relative path
python run_sdar.py usl resnet cifar10 7 1 \
  --bottleneck_mode fixed_random --compression_ratio 8
```

Each folder has a `RUNNING.md` giving the entry points and flags for that
attack. Where a folder also contains a `README.md`, that is the upstream
project's own, kept as-is apart from trimming dataset options the paper does
not use.

`REPRODUCE.md` lists, per table and figure of the paper, the exact command that
produces it and the CSV it writes; `verify_tables.py` compares such a CSV with
the printed numbers (`expected/`).

---

## Datasets

CIFAR-10/100, FashionMNIST, MNIST and GTSRB download automatically via
`torchvision.datasets` / `tf.keras.datasets` on first run.

ImageNet-1k (`utility-imagenet/`) must be provided separately in the standard
train/val directory layout. HAR (`utility-har/`) and
the language-modelling corpus (`utility-llm/`) are fetched by those folders' own
data scripts.

No datasets are included in this artifact.

---

## What is included, and what is not

Included: the defense implementations, the attack code, the experiment runners,
the metric-computation and aggregation scripts, upstream `LICENSE` files, and
`expected/`, the numbers the paper prints, for `verify_tables.py`.

Excluded: datasets, model checkpoints, training traces, reconstruction dumps,
run logs and result tables; `REPRODUCE.md` regenerates them.

Figure and LaTeX-table layout scripts are not included.

## Upstream code

Four of the attack folders are derived from public implementations of the
attacks they run. Each carries the upstream `LICENSE` and an `UPSTREAM.md`
recording the source repository, the base commit, and exactly which files we
modified or added.
