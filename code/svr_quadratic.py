"""SVR with quadratic (degree-2 polynomial) kernel.  Run:  python svr_quadratic.py"""
from sklearn.svm import SVR

from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "SVR_quadratic", scaled_model(SVR(kernel="poly", degree=2, coef0=1)),
        {"C": [0.1, 1, 10], "epsilon": [0.05, 0.1]},
        "Support vector regression, quadratic polynomial kernel.")
