"""SVR with linear kernel.  Run:  python svr_linear.py"""
from sklearn.svm import SVR

from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "SVR_linear", scaled_model(SVR(kernel="linear")),
        {"C": [0.1, 1, 10], "epsilon": [0.05, 0.1, 0.2]},
        "Support vector regression, linear kernel (features and target standardized).")
