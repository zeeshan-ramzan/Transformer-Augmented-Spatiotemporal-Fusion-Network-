"""
Creates a SYNTHETIC dataset with exactly the same columns as the NTHRI file,
so that every script can be tested before the real data are available.

    python make_synthetic_dataset.py                 -> ../dataset/synthetic_nthri_tea_yield.csv
    python make_synthetic_dataset.py --out FILE.csv

Results obtained on synthetic data are meaningless and must not be reported.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def make(n_plots=24, years=range(2015, 2026), months=range(3, 12), keep=0.758, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # shared monthly weather (one on-site station)
    weather = {}
    for y in years:
        for m in range(1, 13):
            season = np.sin((m - 3) / 12 * 2 * np.pi)
            mean_t = 19.6 + 7 * season + rng.normal(0, 1)
            rain = max(0, 60 + 120 * np.exp(-((m - 8) ** 2) / 3) + rng.normal(0, 25))
            weather[(y, m)] = dict(
                min_temp=mean_t - 5.8 + rng.normal(0, 0.4),
                max_temp=mean_t + 5.8 + rng.normal(0, 0.4),
                mean_temp=mean_t,
                humidity=float(np.clip(60 + 0.1 * rain + rng.normal(0, 5), 35, 95)),
                rainfall=rain,
                solar_radiation=196 + 60 * season + rng.normal(0, 15))
    plot_eff = rng.normal(0, 0.06, n_plots)
    for p in range(n_plots):
        for y in years:
            trend = 0.004 * (y - 2015)
            for m in months:
                w = weather[(y, m)]
                prev = weather[(y, m - 1)]
                ndvi = float(np.clip(0.45 + plot_eff[p] + 0.15 * np.sin((m - 3) / 9 * np.pi)
                                     + 0.0004 * prev["rainfall"] + trend + rng.normal(0, 0.02), 0.2, 0.75))
                yld = (120 + 650 * (ndvi - 0.3) + 1.1 * w["rainfall"] + 0.8 * prev["rainfall"]
                       + 6 * (w["min_temp"] - 13) + rng.normal(0, 10))
                gap = int(m in (7, 8, 9) and rng.random() < 0.4)
                row = dict(plot_id=f"P{p + 1:02d}", date=f"{y}-{m:02d}", **w,
                           ndvi_mean=ndvi, ndvi_std=abs(rng.normal(0.07, 0.02)),
                           ndvi_min=ndvi - abs(rng.normal(0.16, 0.03)),
                           ndvi_max=ndvi + abs(rng.normal(0.15, 0.03)),
                           **{"yield": float(np.clip(yld, 85, 1240))},
                           ndvi_gapfilled=gap)
                # optional NDVI sources for the sensitivity analysis
                row.update({k: row[k.replace("l78_", "")] + rng.normal(0, 0.01)
                            for k in C.L78_NDVI_FEATURES})
                if y >= 2017:
                    row.update({k: row[k.replace("s2_", "")] + rng.normal(0, 0.005)
                                for k in C.S2_NDVI_FEATURES})
                else:
                    row.update({k: np.nan for k in C.S2_NDVI_FEATURES})
                rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sample(frac=keep, random_state=seed).sort_values(["plot_id", "date"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.DATASET_DIR / "synthetic_nthri_tea_yield.csv"))
    a = ap.parse_args()
    d = make()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(f"Synthetic dataset written: {a.out}  ({len(d)} rows)")
