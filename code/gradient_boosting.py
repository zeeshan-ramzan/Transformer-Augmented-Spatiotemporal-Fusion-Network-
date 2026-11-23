"""Gradient boosting regression.  Run:  python gradient_boosting.py"""
from sklearn.ensemble import GradientBoostingRegressor

import config as C
from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "Gradient_boosting", scaled_model(GradientBoostingRegressor(random_state=C.RANDOM_STATE), scale_y=False),
        {"n_estimators": [300, 600], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "Gradient boosting with regression trees (squared-error loss).",
        quick_grid={"n_estimators": [50], "learning_rate": [0.1], "max_depth": [3]})
