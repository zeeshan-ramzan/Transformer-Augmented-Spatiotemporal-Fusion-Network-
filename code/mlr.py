"""Multiple linear regression (MLR) baseline.  Run:  python mlr.py"""
from sklearn.linear_model import LinearRegression

from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    run_tabular_algorithm(
        "MLR", scaled_model(LinearRegression(), scale_y=False), {},
        "Multiple linear regression on the 10 standardized features of the target month.")
