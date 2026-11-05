"""Decision-tree regression baseline.  Run:  python decision_tree.py"""
from sklearn.tree import DecisionTreeRegressor

import config as C
from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "Decision_tree", scaled_model(DecisionTreeRegressor(random_state=C.RANDOM_STATE), scale_y=False),
        {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 4, 8]},
        "CART regression tree; depth and leaf size tuned on the inner validation year.")
