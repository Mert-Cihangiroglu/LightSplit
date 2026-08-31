# utility-har — tabular generality (UCI HAR)

Backs the paper's tabular results: LightSplit is not specific to image
classifiers, and the reconstruction picture holds on tabular data.

## Entry points

- `train_har.py` — HAR split training. `--liftback` selects the method
  (`none` = Raw, `fixed` = LightSplit-F, `learned` = LightSplit-L, `codec` =
  learned 1x1, `topk_sparse`), `--k` the payload size on the 256-d cut,
  `--wcc_lambda` the WCC strength.
- `attacks.py --run <dir>` — the three server-side attacks on a trained run:
  membership inference, subject re-identification, input reconstruction.
- `lightsplit.py` — the bottlenecks for the tabular setting; `data.py` downloads
  and caches UCI HAR on first use.

## Reproducing the paper's figures

`reproduce/fig_har.sh` trains and attacks every cell of the three HAR figures
and writes `results/reproduce/fig_har.csv`; `../REPRODUCE.md` gives the
command, its cost and the column meanings.

## Label heterogeneity

`train_noniid.py` trains the same split model on a Dirichlet label partition
over `--num_clients` clients (`--alpha` is the concentration) and writes
`summary.json` with the test accuracy at the best-validation epoch.
`reproduce/tab_har_noniid.sh` runs the paper's sweep; see `REPRODUCE.md`.

`reproduce/tab_har_dp.sh` adds the DP arms (Raw+DP, LightSplit-F+DP) and
emits the HAR DP cells of the privacy overview table; see `../REPRODUCE.md`.
