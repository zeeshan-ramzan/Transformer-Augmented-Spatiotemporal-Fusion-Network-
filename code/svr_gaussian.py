"""SVR with Gaussian (RBF) kernel.  Run:  python svr_gaussian.py"""
from sklearn.svm import SVR

from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "SVR_gaussian", scaled_model(SVR(kernel="rbf")),
        {"C": [1, 10, 100], "gamma": ["scale", 0.05, 0.2], "epsilon": [0.05, 0.1]},
        "Support vector regression, Gaussian (RBF) kernel.")
