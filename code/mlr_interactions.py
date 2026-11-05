"""MLR with pairwise interaction terms.  Run:  python mlr_interactions.py"""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import run_tabular_algorithm

if __name__ == "__main__":
    model = Pipeline([("scaler", StandardScaler()),
                      ("model", Pipeline([("poly", PolynomialFeatures(2, interaction_only=True,
                                                                      include_bias=False)),
                                          ("lr", LinearRegression())]))])
    run_tabular_algorithm(
        "MLR_interactions", model, {},
        "Multiple linear regression with all pairwise interaction terms (45 interactions).")
