"""
Data analysis and preprocessing algorithms of the paper.  Run:  python data_analysis.py

  * descriptive statistics of all features and yield            (Table 2)
  * distribution of observations across years and plots          (Table 3)
  * Pearson correlation analysis (feature-feature and with yield) (Methods)
  * ReliefF (RReliefF for regression) feature ranking and sequential
    forward feature selection, repeated inside every LOYO-CV outer fold
    using only that fold's training years and scored on its inner
    validation year                                              (Methods)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config as C
from common import Reporter, get_args, load_data, loyo_folds, metrics, plt


def rrelieff(X, y, k=10, n_samples=None, seed=C.RANDOM_STATE):
    """RReliefF (Robnik-Sikonja & Kononenko, 2003) feature weights for regression."""
    rng = np.random.default_rng(seed)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    yr = (y - y.min()) / (y.max() - y.min() + 1e-12)
    n, d = X.shape
    idx = rng.choice(n, size=min(n_samples or n, n), replace=False)
    ndc, nda, ndcda = 0.0, np.zeros(d), np.zeros(d)
    for i in idx:
        dist = np.abs(X - X[i]).sum(1); dist[i] = np.inf
        nn = np.argsort(dist)[:k]
        rank = np.arange(1, k + 1); w = np.exp(-(rank / (k / 3)) ** 2); w /= w.sum()
        dy = np.abs(yr[nn] - yr[i])
        dA = np.abs(X[nn] - X[i])
        ndc += np.sum(dy * w)
        nda += (dA * w[:, None]).sum(0)
        ndcda += (dA * (dy * w)[:, None]).sum(0)
    m = len(idx)
    return ndcda / max(ndc, 1e-12) - (nda - ndcda) / max(m - ndc, 1e-12)


def forward_selection(Xtr, ytr, Xva, yva, names):
    """Sequential forward selection with an MLR scorer; returns the RMSE path."""
    sel, rem, path = [], list(range(Xtr.shape[1])), []
    while rem:
        scores = []
        for j in rem:
            cols = sel + [j]
            m = LinearRegression().fit(Xtr[:, cols], ytr)
            scores.append(metrics(yva, m.predict(Xva[:, cols]))["RMSE"])
        b = int(np.argmin(scores))
        sel.append(rem.pop(b))
        path.append((names[sel[-1]], scores[b]))
    best_k = int(np.argmin([p[1] for p in path])) + 1
    return [p[0] for p in path[:best_k]], path


def main():
    args = get_args("Data analysis, correlation, ReliefF and forward feature selection")
    df = load_data(args.data)
    rep = Reporter(args.out, "Data_analysis")
    rep.text(f"Data: {args.data} ({len(df)} observations)")

    rep.section("Descriptive statistics (Table 2)")
    cols = [C.TARGET_COL] + C.FEATURES
    st = df[cols].agg(["min", "max", "mean", "std"]).T.reset_index()
    st.columns = ["feature", "min", "max", "mean", "std_dev"]
    st["CV"] = st["std_dev"] / st["mean"]
    rep.table(st, "table_descriptive_statistics.csv")

    rep.section("Distribution of observations across years and plots (Table 3)")
    ty = df["year"].max()
    t3 = df.groupby("year").agg(plots_monitored=(C.PLOT_COL, "nunique"), months_recorded=("month", "nunique"),
                                observations=(C.TARGET_COL, "size"), mean_yield=(C.TARGET_COL, "mean")).reset_index()
    t3["plots_per_month"] = t3["observations"] / t3["months_recorded"]
    t3["role"] = np.where(t3["year"] == ty, "Independent test year", "Training/LOYO-CV")
    rep.table(t3, "table_observation_distribution.csv", floatfmt="{:.1f}")
    rep.text(f"Total plots {df[C.PLOT_COL].nunique()}, observations {len(df)}, "
             f"observations per plot {len(df) / df[C.PLOT_COL].nunique():.1f}")

    rep.section("Pearson correlation analysis")
    corr = df[cols].corr()
    rep.table(corr.reset_index().rename(columns={"index": "feature"}), "table_correlation_matrix.csv", "{:.3f}")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    lab = ["Yield"] + [C.FEATURE_LABELS[f] for f in C.FEATURES]
    ax.set_xticks(range(len(lab)), lab, rotation=60, ha="right"); ax.set_yticks(range(len(lab)), lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    rep.save_fig(fig, "fig_correlation_matrix.png")

    rep.section("ReliefF ranking and forward selection inside each LOYO-CV outer fold")
    rows, sel_rows = [], []
    X, y = df[C.FEATURES].values, df[C.TARGET_COL].values
    n_samp = 200 if args.quick else None
    for outer, train_years, inner in loyo_folds(df):
        fit = df["year"].isin([t for t in train_years if t != inner]).values
        va = (df["year"] == inner).values
        w = rrelieff(X[fit], y[fit], n_samples=n_samp)
        mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-9
        chosen, path = forward_selection((X[fit] - mu) / sd, y[fit], (X[va] - mu) / sd, y[va], C.FEATURES)
        rows.append({"held_out_year": outer, **{f: wi for f, wi in zip(C.FEATURES, w)}})
        sel_rows.append({"held_out_year": outer, "n_selected": len(chosen), "all_ten_retained": len(chosen) == 10,
                         "selected": ", ".join(chosen)})
    rw = pd.DataFrame(rows)
    rep.table(rw, "table_relieff_weights_per_fold.csv")
    rep.table(pd.DataFrame(sel_rows), "table_forward_selection_per_fold.csv")
    n_all = int(sum(r["all_ten_retained"] for r in sel_rows))
    rep.text(f"All ten features retained in {n_all} of {len(sel_rows)} outer folds; "
             f"positive ReliefF weight for all features in "
             f"{int((rw[C.FEATURES] > 0).all(1).sum())} of {len(rw)} folds.")
    mw = rw[C.FEATURES].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([C.FEATURE_LABELS[f] for f in mw.index], mw.values, color="#3b7dc4", ec="k", lw=0.4)
    ax.set_xlabel("Mean ReliefF weight over LOYO-CV folds")
    rep.save_fig(fig, "fig_relieff_weights.png")
    rep.json.update({"n_folds_all_ten_retained": n_all, "relieff_mean": mw.to_dict()})
    rep.finish()


if __name__ == "__main__":
    main()
