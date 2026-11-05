"""XGBoost regression.  Run:  python xgboost_model.py"""
from xgboost import XGBRegressor

import config as C
from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "XGBoost", scaled_model(XGBRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, verbosity=0), scale_y=False),
        {"n_estimators": [300, 600], "learning_rate": [0.05, 0.1], "max_depth": [4, 6],
         "subsample": [0.8], "colsample_bytree": [0.8]},
        "Extreme gradient boosting (XGBoost).",
        quick_grid={"n_estimators": [50], "learning_rate": [0.1], "max_depth": [4]})
