"""
Collects the results of all algorithms (run each algorithm file first) and
produces the overall comparison (Table 7) and the bar chart of the best
methods (Fig. 2).  Run:  python compare_all.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import Reporter, get_args, plt

ORDER = [("Conventional regression", ["MLR", "MLR_interactions", "Stepwise_MLR", "Decision_tree",
                                      "SVR_linear", "SVR_quadratic", "SVR_cubic", "SVR_gaussian"]),
         ("GPR variants", ["GPR_rational_quadratic", "GPR_squared_exponential", "GPR_matern52", "GPR_exponential"]),
         ("Ensemble methods", ["Random_forest", "Gradient_boosting", "XGBoost"]),
         ("NAS-DNN", ["NAS_DNN"]), ("Proposed", ["TASTFN"])]


def main():
    args = get_args("Comparison of all algorithms (Table 7, Fig. 2)")
    root = Path(args.out)
    rows = []
    for cat, names in ORDER:
        for n in names:
            f = root / n / "summary.json"
            if not f.exists():
                print(f"  (missing results for {n}; run its script first)")
                continue
            s = json.loads(f.read_text())
            L, T = s["loyo"], s["test"] or {}
            rows.append({"category": cat, "model": n,
                         "R2": L["R2"], "R2_sd": L["R2_sd"], "RMSE": L["RMSE"], "RMSE_sd": L["RMSE_sd"],
                         "MAE": L["MAE"], "MAE_sd": L["MAE_sd"], "MAPE": L["MAPE"], "MAPE_sd": L["MAPE_sd"],
                         "test_R2": T.get("R2", np.nan), "test_RMSE": T.get("RMSE", np.nan),
                         "test_MAPE": T.get("MAPE", np.nan),
                         "fold_RMSE": [r["RMSE"] for r in s["loyo_folds"]]})
    if not rows:
        raise SystemExit("No results found. Run the algorithm scripts first.")
    df = pd.DataFrame(rows)
    rep = Reporter(args.out, "Comparison")
    rep.section("All methods, LOYO-CV mean ± SD and independent test year (Table 7)")
    rep.table(df.drop(columns="fold_RMSE"), "table_all_methods.csv")

    best = {k: df.loc[df[k].idxmin() if k != "R2" and k != "test_R2" else df[k].idxmax(), "model"]
            for k in ["R2", "RMSE", "MAE", "MAPE", "test_R2", "test_RMSE", "test_MAPE"]}
    rep.text("Best model per metric: " + ", ".join(f"{k}: {v}" for k, v in best.items()))
    if "TASTFN" in df.model.values and len(df) > 1:
        base = df[df.model != "TASTFN"].sort_values("RMSE").iloc[0]
        t = df[df.model == "TASTFN"].iloc[0]
        rep.text(f"Strongest baseline: {base.model} (LOYO-CV RMSE {base.RMSE:.2f}); "
                 f"TASTFN RMSE {t.RMSE:.2f} -> reduction {100 * (1 - t.RMSE / base.RMSE):.1f}% "
                 f"(test year: {100 * (1 - t.test_RMSE / base.test_RMSE):.1f}%)")
        try:
            from scipy.stats import wilcoxon
            a, b = np.array(t.fold_RMSE), np.array(base.fold_RMSE)
            if len(a) == len(b) and len(a) > 1:
                w = wilcoxon(a, b)
                rep.text(f"Wilcoxon signed-rank test on fold RMSE (TASTFN vs {base.model}): "
                         f"p = {w.pvalue:.4f}; TASTFN lower in {int((a < b).sum())} of {len(a)} folds")
        except Exception as e:
            rep.text(f"(Wilcoxon test not computed: {e})")

    top = df[df.model != "TASTFN"].sort_values("RMSE").head(5).iloc[::-1]
    top = pd.concat([top, df[df.model == "TASTFN"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(top))
    axes[0].bar(x - 0.2, top.RMSE, 0.4, yerr=top.RMSE_sd, capsize=3, color="#8fb3d9", ec="k", lw=0.4, label="RMSE")
    axes[0].bar(x + 0.2, top.MAE, 0.4, yerr=top.MAE_sd, capsize=3, color="#cdd7e0", ec="k", lw=0.4,
                hatch="//", label="MAE")
    axes[0].set_ylabel("Error (kg ha$^{-1}$)"); axes[0].legend(); axes[0].set_title("(a) RMSE and MAE", loc="left")
    axes[1].bar(x, top.MAPE, 0.6, yerr=top.MAPE_sd, capsize=3, color="#8fb3d9", ec="k", lw=0.4)
    axes[1].set_ylabel("MAPE (%)"); axes[1].set_title("(b) MAPE", loc="left")
    for ax in axes:
        ax.set_xticks(x, top.model.str.replace("_", " "), rotation=30, ha="right")
    rep.save_fig(fig, "fig_comparison_best_methods.png")
    rep.finish()


if __name__ == "__main__":
    main()
