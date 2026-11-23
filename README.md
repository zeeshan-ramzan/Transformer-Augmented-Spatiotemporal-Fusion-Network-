# TASTFN replication code

Code to reproduce results of *"A transformer-based spatiotemporal fusion network for farm-level tea yield estimation using Landsat-8 NDVI and agrometeorological data"*: the proposed TASTFN, the sixteen baseline methods, and the data-analysis algorithms. Each algorithm is a separate file.

## Folder layout

```
TASTFN_Replication/
├── code/                      <- all code (run everything from here)
│   ├── config.py              paths, column names, protocol, TASTFN hyperparameters
│   ├── common.py              data loading, splits, metrics, plots, shared report
│   ├── tastfn.py              PROPOSED ARCHITECTURE (all TASTFN analyses)
│   ├── mlr.py  mlr_interactions.py  stepwise_mlr.py  decision_tree.py
│   ├── svr_linear.py  svr_quadratic.py  svr_cubic.py  svr_gaussian.py
│   ├── gpr_rational_quadratic.py  gpr_squared_exponential.py
│   ├── gpr_matern52.py  gpr_exponential.py
│   ├── random_forest.py  gradient_boosting.py  xgboost_model.py
│   ├── nas_dnn.py             NAS-DNN (Bayesian-optimized architecture search)
│   ├── data_analysis.py       Table 2, Table 3, correlation, ReliefF + forward selection
│   ├── compare_all.py         Table 7 and Fig. 2 from the saved results
│   ├── run_all.py             runs everything in sequence
│   └── make_synthetic_dataset.py   test data with the same columns
├── dataset/
│   ├── README.txt             required column names
│   └── nthri_tea_yield.csv    <- PUT YOUR DATA HERE
├── results/                   created automatically, one folder per algorithm
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is used automatically by `tastfn.py` and `nas_dnn.py` when available.

## Running

```bash
cd code
python tastfn.py              # proposed method
python mlr.py                 # any baseline, one at a time
python data_analysis.py
python compare_all.py         # after the algorithms have run
python run_all.py             # or everything in one go
```

### Useful options

```bash
python tastfn.py --data ../dataset/my_data.xlsx   # explicit dataset file
python tastfn.py --quick                          # fast smoke test, not for reporting
python tastfn.py --hp-protocol inner              # hyperparameter sensitivity on a
                                                  #   temporally blocked split (default)
python tastfn.py --hp-protocol stratified         # ... under stratified 10-fold CV
```

`--hp-protocol` selects the protocol for the hyperparameter sensitivity grid
only. It does not affect the leave-one-year-out or test-year results, whose
configuration is fixed in `config.TASTFN_PARAMS`.

### Evaluation protocol in one paragraph

The independent test year (the last year present, 2025) is set aside before
anything else. Leave-one-year-out cross-validation runs over the remaining
years: each year in turn is the outer test fold, the most recent remaining
year is the inner validation year for early stopping and selection, and the
scaler is fitted on the training years only. Because consecutive six-month
windows of one plot overlap in five months, blocking is applied position by
position inside each window: every context position falling in the held-out
year is masked out of the training and inner-validation tensors and excluded
from attention, so no training window carries any observation of the held-out
year. Stratified 10-fold cross-validation is also reported, as an optimistic
upper bound only. See `SequenceData.with_context` / `.tensors` and `run_loyo`
in `code/tastfn.py`.
Options for every script: `--data PATH` (default: first match in `config.DATA_FILE_CANDIDATES`),
`--out FOLDER` (default `../results`), `--quick` (fast smoke test, not for reporting).
`tastfn.py` also accepts `--skip-ablation --skip-importance --skip-sensitivity
--skip-variants --skip-hparam --hp-protocol {inner,stratified} --n-perm N`.

## Train / validation / test split (as in the paper)

| Design                                         | Training                                   | Model selection / early stopping       | Testing                     |
| ---------------------------------------------- | ------------------------------------------ | -------------------------------------- | --------------------------- |
| **Independent test year**                      | all years before the last year (2015–2024) | most recent training year (2024)       | last year (2025), used once |
| **Leave-one-year-out CV** (primary)            | 9 of the 10 years 2015–2024                | most recent training year of each fold | the held-out year           |
| Stratified 10-fold CV (continuity, optimistic) | 9/10 of observations (yield-decile strata) | 10% of the training fold (TASTFN)      | remaining 1/10              |

* Standardization is fitted on the training years of each fold only.
* Feature selection (ReliefF + forward selection, `data_analysis.py`) is repeated inside each
  outer fold using only training years and scored on the inner validation year.
* Baseline hyperparameters are tuned on the inner validation year; tabular models are then
  refitted on all training years of the fold. Neural networks keep the early-stopped weights
  (TASTFN) or are refitted for the selected number of epochs (NAS-DNN).
* TASTFN windows never contain inputs from the held-out year; windows of the test year may
  use earlier months, which would be available in practice.
* All metrics (R², RMSE, MSE, MAE, MAPE) are computed on the original kg ha⁻¹ scale from the
  same predictions. Residuals are defined as **predicted − observed**.

## Testing reported by each script

Every algorithm writes to `results/<Algorithm>/`:

* `summary.txt` – all tables in readable form; `summary.json` – machine-readable
* `table_evaluation_designs.csv` (Table 5), `table_loyo_per_year.csv` (Table 6),
  `table_errors_by_yield_range.csv` (Table 9), `table_residual_distribution.csv` (Table 10),
  `table_seasonal.csv` (Table 14), predictions for LOYO-CV / test year / stratified CV
* `figures/` – observed vs predicted scatter, histograms, residuals vs index, residual
  histogram, RMSE per held-out year

`tastfn.py` additionally produces learning curves (Fig. 5), attention heat map and gate values
(Fig. 6), ablation (Table 11), permutation importance (Table 12, Fig. 7), seasonal vs plot-level
skill (Table 13), NDVI-source sensitivity (Table 15), hyperparameter sensitivity (Table 16),
log-target and forecast-mode variants (Table 17), computational cost (Table 18) and the saved model.
`compare_all.py` produces Table 7, Fig. 2, the reduction relative to the strongest baseline and a
Wilcoxon signed-rank test on fold RMSE.

## Run time

On a CPU the full `tastfn.py` run (≈200 network fits) takes several hours; on an RTX-class GPU
it is much faster. Use the `--skip-*` options to run parts separately. GPR files take a few
minutes each because of kernel optimization.
