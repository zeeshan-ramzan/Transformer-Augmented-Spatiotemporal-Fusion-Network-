"""SVR with cubic (degree-3 polynomial) kernel.  Run:  python svr_cubic.py"""
from sklearn.svm import SVR

from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "SVR_cubic", scaled_model(SVR(kernel="poly", degree=3, coef0=1)),
        {"C": [0.1, 1, 10], "epsilon": [0.05, 0.1]},
        "Support vector regression, cubic polynomial kernel.")
