"""
Shared utilities: data loading, evaluation splits, metrics, reporting, plots,
and a generic runner used by every non-sequential (tabular) baseline.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config as C  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


# =============================================================================
# Command line
# =============================================================================
def get_args(description: str, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=str(C.DATA_FILE),
                   help="CSV file (default: ../dataset/nthri_tea_yield.csv)")
    p.add_argument("--out", default=str(C.RESULTS_DIR),
                   help="results root folder (default: ../results)")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke-test settings (not for reporting)")
    if extra:
        extra(p)
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================
def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            f"Place the CSV at {C.DATA_FILE} (see dataset/README.txt),\n"
            f"or run  python make_synthetic_dataset.py  to create a test file.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if C.PLOT_COL not in df.columns and "farm_id" in df.columns:
        df = df.rename(columns={"farm_id": C.PLOT_COL})
    missing = [c for c in [C.PLOT_COL, C.DATE_COL, C.TARGET_COL] + C.FEATURES
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL].astype(str))
    df["year"] = df[C.DATE_COL].dt.year
    df["month"] = df[C.DATE_COL].dt.month
    df = df.dropna(subset=C.FEATURES + [C.TARGET_COL])
    df = df.sort_values([C.PLOT_COL, C.DATE_COL]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def test_year(df) -> int:
    return int(C.TEST_YEAR) if C.TEST_YEAR is not None else int(df["year"].max())


def cv_years(df) -> list[int]:
    ty = test_year(df)
    return sorted(int(y) for y in df["year"].unique() if y < ty)


def loyo_folds(df):
    """Yield (outer_year, train_years, inner_val_year) for LOYO-CV."""
    years = cv_years(df)
    for y in years:
        train_years = [t for t in years if t != y]
        yield y, train_years, max(train_years)


def stratified_folds(df, n_splits=C.N_STRATIFIED_FOLDS, seed=C.RANDOM_STATE):
    """Stratified K-fold on yield deciles (random assignment, all years)."""
    deciles = pd.qcut(df[C.TARGET_COL].rank(method="first"), 10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(df)), deciles))


# =============================================================================
# Metrics
# =============================================================================
def metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = p - y
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)),
            "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
            "RMSE": float(np.sqrt(mse)), "MSE": mse,
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(100 * np.mean(np.abs(err / y)))}


def mean_sd(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {}
    for k in ["R2", "RMSE", "MSE", "MAE", "MAPE"]:
        out[k] = float(df[k].mean())
        out[k + "_sd"] = float(df[k].std(ddof=1)) if len(df) > 1 else 0.0
    out["n"] = int(df["n"].sum())
    return out


def error_distribution(y, p) -> dict:
    """Residuals defined as predicted minus observed (kg/ha)."""
    r = np.asarray(p, float) - np.asarray(y, float)
    a = np.abs(r)
    rmse = float(np.sqrt(np.mean(r ** 2)))
    mae = float(np.mean(a))
    return {"mean": float(r.mean()), "median": float(np.median(r)),
            "sd": float(r.std(ddof=1)),
            "p5": float(np.percentile(r, 5)), "p25": float(np.percentile(r, 25)),
            "p75": float(np.percentile(r, 75)), "p95": float(np.percentile(r, 95)),
            "max_abs": float(a.max()), "skewness": float(stats.skew(r)),
            "excess_kurtosis": float(stats.kurtosis(r)),
            "rmse_mae_ratio": rmse / mae if mae > 0 else np.nan,
            "pct_gt10": float(100 * np.mean(a > 10)),
            "pct_gt25": float(100 * np.mean(a > 25)),
            "pct_gt50": float(100 * np.mean(a > 50)),
            "within_1.96sd": float(100 * np.mean(np.abs(r - r.mean()) <= 1.96 * r.std(ddof=1)))}


def yield_range_table(pred_df, cuts) -> pd.DataFrame:
    """Errors by yield class; cuts = (low_upper, high_lower) tertiles."""
    y, p = pred_df["observed"].values, pred_df["predicted"].values
    lo, hi = cuts
    classes = {f"Low (<{lo:.0f})": y < lo,
               f"Medium ({lo:.0f}-{hi:.0f})": (y >= lo) & (y <= hi),
               f"High (>{hi:.0f})": y > hi}
    rows = []
    for name, m in classes.items():
        if m.sum() == 0:
            continue
        mm = metrics(y[m], p[m])
        rows.append({"class": name, "n": mm["n"], "RMSE": mm["RMSE"], "MAE": mm["MAE"],
                     "MAPE": mm["MAPE"], "Bias(pred-obs)": float(np.mean(p[m] - y[m]))})
    return pd.DataFrame(rows)


def seasonal_table(pred_df) -> pd.DataFrame:
    rows = []
    for name, months in C.SEASONS.items():
        m = pred_df["month"].isin(months).values
        if m.sum() < 2:
            continue
        mm = metrics(pred_df.loc[m, "observed"], pred_df.loc[m, "predicted"])
        rows.append({"season": name, **{k: mm[k] for k in ["n", "R2", "RMSE", "MAE", "MAPE"]}})
    mm = metrics(pred_df["observed"], pred_df["predicted"])
    rows.append({"season": "Overall", **{k: mm[k] for k in ["n", "R2", "RMSE", "MAE", "MAPE"]}})
    return pd.DataFrame(rows)


def between_plot_r2(pred_df) -> float:
    """R2 of observed vs predicted deviations from the year-month mean of all plots."""
    d = pred_df.copy()
    g = d.groupby(["year", "month"])
    d["obs_dev"] = d["observed"] - g["observed"].transform("mean")
    d["pred_dev"] = d["predicted"] - g["predicted"].transform("mean")
    ss_tot = np.sum(d["obs_dev"] ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - np.sum((d["obs_dev"] - d["pred_dev"]) ** 2) / ss_tot)


# =============================================================================
# Output helpers
# =============================================================================
class Reporter:
    """Collects text, tables and figures of one algorithm in results/<name>/."""

    def __init__(self, out_root, name):
        self.dir = Path(out_root) / name
        self.fig_dir = self.dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.lines = [f"RESULTS: {name}", "=" * 78,
                      f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        self.json = {"algorithm": name}

    def section(self, title):
        self.lines += ["", title, "-" * len(title)]

    def text(self, s):
        self.lines.append(str(s))
        print(s, flush=True)

    def table(self, df: pd.DataFrame, fname=None, floatfmt="{:.4f}"):
        txt = df.to_string(index=False, float_format=lambda v: floatfmt.format(v))
        self.lines.append(txt)
        print(txt, flush=True)
        if fname:
            df.to_csv(self.dir / fname, index=False)

    def save_fig(self, fig, fname):
        fig.tight_layout()
        fig.savefig(self.fig_dir / fname, bbox_inches="tight")
        plt.close(fig)
        self.lines.append(f"[figure] figures/{fname}")

    def finish(self):
        (self.dir / "summary.txt").write_text("\n".join(self.lines), encoding="utf-8")
        with open(self.dir / "summary.json", "w") as f:
            json.dump(self.json, f, indent=2, default=float)
        print(f"\nAll results written to: {self.dir}", flush=True)


def fmt_pm(m, k, nd=4):
    return f"{m[k]:.{nd}f} ± {m[k + '_sd']:.{nd}f}"


def summary_table(results: dict) -> pd.DataFrame:
    """results: {'design': metrics-dict}; mean±SD columns where available."""
    rows = []
    for design, m in results.items():
        row = {"Evaluation design": design, "n": m.get("n")}
        for k, nd in [("R2", 4), ("RMSE", 2), ("MAE", 2), ("MAPE", 2)]:
            row[k] = fmt_pm(m, k, nd) if k + "_sd" in m else f"{m[k]:.{nd}f}"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================
def plot_scatter(rep, loyo_pred, test_pred, fname="fig_scatter_observed_vs_predicted.png"):
    y, p = loyo_pred["observed"].values, loyo_pred["predicted"].values
    slope, intercept = np.polyfit(y, p, 1)
    r2 = metrics(y, p)["R2"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [min(y.min(), p.min()) * 0.95, max(y.max(), p.max()) * 1.03]
    ax.plot(lim, lim, "--", color="#555", lw=1, label="Identity line")
    ax.scatter(y, p, s=9, alpha=0.5, color="#3b7dc4", label=f"LOYO-CV records (n = {len(y)})")
    if test_pred is not None and len(test_pred):
        ax.scatter(test_pred["observed"], test_pred["predicted"], s=12, alpha=0.8,
                   color="#e39a2d", label=f"Test year (n = {len(test_pred)})")
    xs = np.array(lim)
    ax.plot(xs, slope * xs + intercept, color="#d0453a", lw=1.5,
            label=f"Best fit (slope = {slope:.4f})")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed yield (kg ha$^{-1}$)"); ax.set_ylabel("Predicted yield (kg ha$^{-1}$)")
    ax.text(0.97, 0.04, f"$R^2$ = {r2:.4f}", transform=ax.transAxes, ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="#aaa"))
    ax.legend(loc="upper left", fontsize=8)
    rep.save_fig(fig, fname)
    return float(slope), float(intercept)


def plot_histogram(rep, pred_df, fname="fig_histogram_observed_predicted.png"):
    y, p = pred_df["observed"].values, pred_df["predicted"].values
    edges = np.arange(0, max(y.max(), p.max()) + 100, 100)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ho, _ = np.histogram(y, edges); hp, _ = np.histogram(p, edges)
    c = edges[:-1] + 50
    ax.bar(c - 20, ho, width=40, color="#3b7dc4", ec="k", lw=0.4, label=f"Observed (n = {len(y)})")
    ax.bar(c + 20, hp, width=40, color="#d0453a", ec="k", lw=0.4, label=f"Predicted (n = {len(p)})")
    ax.set_xticks(edges); ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Yield (kg ha$^{-1}$), bin width 100"); ax.set_ylabel("Frequency")
    ax.legend()
    rep.save_fig(fig, fname)


def plot_residuals(rep, pred_df, fname="fig_residuals_vs_index.png"):
    d = pred_df.sort_values(["year", "plot_id", "month"]).reset_index(drop=True)
    r = (d["predicted"] - d["observed"]).values
    sd = r.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axhspan(-1.96 * sd, 1.96 * sd, color="#cfe8d5", alpha=0.6,
               label=f"±1.96σ (±{1.96 * sd:.1f} kg ha$^{{-1}}$)")
    ax.scatter(np.arange(len(r)), r, s=6, alpha=0.6, color="#3b7dc4")
    ax.axhline(0, color="#d0453a", ls="--", lw=1)
    for y, idx in d.groupby("year").groups.items():
        ax.axvline(min(idx), color="#999", lw=0.6)
        ax.text(np.mean(idx), ax.get_ylim()[1] * 0.92 if len(r) else 0, str(y),
                ha="center", fontsize=7, color="#555")
    ax.set_xlabel("Observation index (pooled, ordered by year)")
    ax.set_ylabel("Residual, predicted − observed (kg ha$^{-1}$)")
    ax.set_title(f"n = {len(r)}   mean {r.mean():.2f}   SD {sd:.2f}   max |error| {np.abs(r).max():.2f}",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    rep.save_fig(fig, fname)


def plot_residual_hist(rep, pred_df, fname="fig_residual_histogram.png"):
    r = (pred_df["predicted"] - pred_df["observed"]).values
    fig, ax = plt.subplots(figsize=(6, 4))
    bw = max(1.0, round(2 * stats.iqr(r) / len(r) ** (1 / 3)))
    bins = np.arange(np.floor(r.min() / bw) * bw, r.max() + bw, bw)
    ax.hist(r, bins=bins, color="#3b7dc4", ec="k", lw=0.4, label=f"Residuals (n = {len(r)})")
    xs = np.linspace(bins[0], bins[-1], 300)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std(ddof=1)) * len(r) * bw, color="#d0453a",
            lw=2, label=f"Normal (mean {r.mean():.2f}, SD {r.std(ddof=1):.2f})")
    ax.set_xlabel("Residual, predicted − observed (kg ha$^{-1}$)"); ax.set_ylabel("Frequency")
    ax.text(0.02, 0.95, f"skewness {stats.skew(r):.2f}\nexcess kurtosis {stats.kurtosis(r):.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#aaa"))
    ax.legend(loc="upper right", fontsize=8)
    rep.save_fig(fig, fname)


def plot_per_year(rep, fold_rows, test_m=None, fname="fig_rmse_per_year.png"):
    df = pd.DataFrame(fold_rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(df["year"].astype(str), df["RMSE"], color="#3b7dc4", ec="k", lw=0.4, label="LOYO-CV fold")
    if test_m is not None:
        ax.bar([str(test_m["year"])], [test_m["RMSE"]], color="#e39a2d", ec="k", lw=0.4,
               label="Independent test year")
    ax.axhline(df["RMSE"].mean(), color="#d0453a", ls="--", lw=1, label="LOYO-CV mean")
    ax.set_ylabel("RMSE (kg ha$^{-1}$)"); ax.set_xlabel("Held-out year")
    ax.legend(fontsize=8)
    rep.save_fig(fig, fname)


# =============================================================================
# Full standard report (shared by all algorithms)
# =============================================================================
def standard_report(rep, df, loyo_rows, loyo_pred, test_pred, strat_rows, strat_pred,
                    extra_designs=None):
    """Writes the tables/figures common to every algorithm.

    loyo_rows : list of per-fold metric dicts (with 'year')
    loyo_pred / test_pred / strat_pred : DataFrames with observed, predicted,
                year, month, plot_id
    """
    ty = test_year(df)
    loyo_m = mean_sd(loyo_rows)
    pooled_m = metrics(loyo_pred["observed"], loyo_pred["predicted"])
    test_m = metrics(test_pred["observed"], test_pred["predicted"]) if len(test_pred) else None
    strat_m = mean_sd(strat_rows) if strat_rows else None

    rep.section("Performance under the three evaluation designs (Table 5 format)")
    designs = {}
    if strat_m:
        designs["Stratified 10-fold CV (random split, all years; optimistic)"] = strat_m
    designs[f"LOYO-CV {min(cv_years(df))}-{max(cv_years(df))} (fold mean ± SD)"] = loyo_m
    designs[f"LOYO-CV {min(cv_years(df))}-{max(cv_years(df))} (pooled)"] = pooled_m
    if test_m:
        designs[f"Independent test year ({ty})"] = test_m
    if extra_designs:
        designs.update(extra_designs)
    rep.table(summary_table(designs), "table_evaluation_designs.csv")

    rep.section("LOYO-CV results per held-out year (Table 6 format)")
    fy = pd.DataFrame(loyo_rows)[["year", "n", "R2", "RMSE", "MAE", "MAPE"]]
    rep.table(fy, "table_loyo_per_year.csv")

    tert = tuple(np.percentile(loyo_pred["observed"], [100 / 3, 200 / 3]))
    rep.section(f"Errors by yield range, tertiles {tert[0]:.0f} / {tert[1]:.0f} kg/ha (Table 9 format)")
    yr = yield_range_table(loyo_pred, tert); yr.insert(0, "data", "LOYO-CV")
    if test_m:
        yt = yield_range_table(test_pred, tert); yt.insert(0, "data", f"Test {ty}")
        yr = pd.concat([yr, yt], ignore_index=True)
    rep.table(yr, "table_errors_by_yield_range.csv")

    rep.section("Residual distribution, predicted − observed (Table 10 format)")
    ed = {"LOYO-CV (pooled)": error_distribution(loyo_pred["observed"], loyo_pred["predicted"])}
    if test_m:
        ed[f"Test {ty}"] = error_distribution(test_pred["observed"], test_pred["predicted"])
    rep.table(pd.DataFrame(ed).reset_index().rename(columns={"index": "statistic"}),
              "table_residual_distribution.csv")

    rep.section("Seasonal performance, pooled LOYO-CV predictions (Table 14 format)")
    rep.table(seasonal_table(loyo_pred), "table_seasonal.csv")

    rep.section("Between-plot skill")
    bp = between_plot_r2(loyo_pred)
    rep.text(f"Between-plot R2 (deviations from year-month mean of all plots): {bp:.4f}")

    # Figures
    slope, intercept = plot_scatter(rep, loyo_pred, test_pred)
    rep.text(f"Best-fit line (pooled LOYO-CV): slope {slope:.4f}, intercept {intercept:.2f} kg/ha")
    plot_histogram(rep, loyo_pred)
    plot_residuals(rep, loyo_pred)
    plot_residual_hist(rep, loyo_pred)
    plot_per_year(rep, loyo_rows, {"year": ty, **test_m} if test_m else None)

    # Predictions
    loyo_pred.to_csv(rep.dir / "predictions_loyo.csv", index=False)
    test_pred.to_csv(rep.dir / "predictions_test.csv", index=False)
    if strat_pred is not None:
        strat_pred.to_csv(rep.dir / "predictions_stratified.csv", index=False)

    rep.json.update({"loyo": loyo_m, "loyo_pooled": pooled_m, "test": test_m,
                     "stratified": strat_m, "loyo_folds": loyo_rows,
                     "between_plot_r2": bp, "slope": slope, "intercept": intercept,
                     "tertiles": list(tert)})
    return loyo_m, pooled_m, test_m, strat_m


# =============================================================================
# Generic runner for tabular (non-sequential) models
# =============================================================================
def scaled_model(estimator, scale_y=True):
    """Standardization fitted on training rows only (inside fit)."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    if scale_y:
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    return pipe


def _grid(param_grid):
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[param_grid[k] for k in keys])]


def _set(model, params):
    m = clone(model)
    if params:
        if isinstance(m, TransformedTargetRegressor):
            prefix = "regressor__model__"
        elif isinstance(m, Pipeline):
            prefix = "model__"
        else:
            prefix = ""
        m.set_params(**{prefix + k: v for k, v in params.items()})
    return m


def tune(model, grid, Xtr, ytr, Xva, yva):
    """Select hyperparameters on the inner validation year (lowest RMSE)."""
    best, best_rmse = {}, np.inf
    for params in grid:
        m = _set(model, params)
        m.fit(Xtr, ytr)
        r = metrics(yva, m.predict(Xva))["RMSE"]
        if r < best_rmse:
            best, best_rmse = params, r
    return best, best_rmse


def _pred_frame(d, pred):
    return pd.DataFrame({"plot_id": d[C.PLOT_COL].values, "year": d["year"].values,
                         "month": d["month"].values, "observed": d[C.TARGET_COL].values,
                         "predicted": np.asarray(pred, float)})


def run_tabular_algorithm(name, model, param_grid, description, quick_grid=None, tuner=None):
    """Full evaluation of one tabular baseline (LOYO-CV, test year, stratified CV).

    tuner(model, Xtr, ytr, Xva, yva, quick) -> params  replaces the grid search
    (used by the NAS-DNN, whose architecture is found by Bayesian optimization).
    """
    args = get_args(description)
    df = load_data(args.data)
    grid = _grid(quick_grid if (args.quick and quick_grid is not None) else param_grid)
    if args.quick:
        grid = grid[:1]
    rep = Reporter(args.out, name)
    rep.text(description)
    rep.text(f"Data: {args.data}  ({len(df)} observations, years {df.year.min()}-{df.year.max()})")
    rep.text(f"Features ({len(C.FEATURES)}): {', '.join(C.FEATURES)}")
    if tuner is None:
        rep.text(f"Hyperparameter grid ({len(grid)} combinations), selected on the inner validation year")
    else:
        rep.text("Hyperparameters/architecture selected on the inner validation year by Bayesian optimization")
    X, y = df[C.FEATURES].values, df[C.TARGET_COL].values
    ty = test_year(df)

    t0 = time.time()
    loyo_rows, loyo_preds, chosen = [], [], []
    for outer, train_years, inner in loyo_folds(df):
        tr = df["year"].isin(train_years).values
        fit_m = df["year"].isin([t for t in train_years if t != inner]).values
        va = (df["year"] == inner).values
        te = (df["year"] == outer).values
        if tuner is not None:
            params = tuner(model, X[fit_m], y[fit_m], X[va], y[va], args.quick)
        else:
            params, _ = tune(model, grid, X[fit_m], y[fit_m], X[va], y[va])
        m = _set(model, params).fit(X[tr], y[tr])      # refit on all training years
        p = m.predict(X[te])
        loyo_rows.append({"year": outer, **metrics(y[te], p)})
        loyo_preds.append(_pred_frame(df[te], p))
        chosen.append({"held_out_year": outer, "inner_val_year": inner, **params})
        rep.text(f"  LOYO fold {outer}: RMSE {loyo_rows[-1]['RMSE']:.2f}  params {params}")
    loyo_time = time.time() - t0
    loyo_pred = pd.concat(loyo_preds, ignore_index=True)

    # Independent test year: train on all CV years, tune on the most recent one
    years = cv_years(df)
    tr = df["year"].isin(years).values
    fit_m = df["year"].isin(years[:-1]).values
    va = (df["year"] == years[-1]).values
    te = (df["year"] == ty).values
    if tuner is not None:
        test_params = tuner(model, X[fit_m], y[fit_m], X[va], y[va], args.quick)
    else:
        test_params, _ = tune(model, grid, X[fit_m], y[fit_m], X[va], y[va])
    final = _set(model, test_params).fit(X[tr], y[tr])
    t1 = time.time(); p_test = final.predict(X[te]); infer_t = time.time() - t1
    test_pred = _pred_frame(df[te], p_test)

    # Stratified 10-fold CV (uses the test-model hyperparameters)
    strat_rows, strat_preds = [], []
    folds = stratified_folds(df)
    if args.quick:
        folds = folds[:3]
    for k, (a, b) in enumerate(folds):
        m = _set(model, test_params).fit(X[a], y[a])
        p = m.predict(X[b])
        strat_rows.append({"fold": k + 1, **metrics(y[b], p)})
        strat_preds.append(_pred_frame(df.iloc[b], p))
    strat_pred = pd.concat(strat_preds, ignore_index=True)

    rep.section("Selected hyperparameters per LOYO fold")
    rep.table(pd.DataFrame(chosen), "table_selected_hyperparameters.csv")
    rep.text(f"Test-year model hyperparameters: {test_params}")
    standard_report(rep, df, loyo_rows, loyo_pred, test_pred, strat_rows, strat_pred)
    rep.section("Computational cost")
    ops = len(p_test) / infer_t if infer_t > 0 else float("nan")
    rep.text(f"LOYO-CV run time: {loyo_time:.1f} s   inference: {ops:,.0f} observations/s")
    rep.json.update({"loyo_time_s": loyo_time, "inference_obs_per_s": ops,
                     "test_hyperparameters": test_params})
    rep.finish()
    return rep
