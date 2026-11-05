"""Random forest regression.  Run:  python random_forest.py"""
from sklearn.ensemble import RandomForestRegressor

import config as C
from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "Random_forest", scaled_model(RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1), scale_y=False),
        {"n_estimators": [300, 600], "max_features": [1.0, 0.5], "min_samples_leaf": [1, 3]},
        "Random forest (bagged regression trees).",
        quick_grid={"n_estimators": [50], "max_features": [1.0], "min_samples_leaf": [1]})
