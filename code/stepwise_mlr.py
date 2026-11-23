"""
Stepwise multiple linear regression (bidirectional: forward entry and backward
removal based on t-test p-values).  Run:  python stepwise_mlr.py
"""
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin

from common import run_tabular_algorithm, scaled_model


class StepwiseLinearRegression(BaseEstimator, RegressorMixin):
    """Bidirectional stepwise OLS.  A predictor enters when its p-value < p_enter
    and is removed when its p-value rises above p_remove."""

    def __init__(self, p_enter=0.05, p_remove=0.10, max_iter=100):
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.max_iter = max_iter

    @staticmethod
    def _ols(X, y):
        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        dof = max(len(y) - A.shape[1], 1)
        sigma2 = resid @ resid / dof
        cov = sigma2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        p = 2 * stats.t.sf(np.abs(beta / se), dof)
        return beta, p

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        selected, remaining = [], list(range(X.shape[1]))
        for _ in range(self.max_iter):
            changed = False
            # forward step
            best_p, best_j = 1.0, None
            for j in remaining:
                _, p = self._ols(X[:, selected + [j]], y)
                if p[-1] < best_p:
                    best_p, best_j = p[-1], j
            if best_j is not None and best_p < self.p_enter:
                selected.append(best_j); remaining.remove(best_j); changed = True
            # backward step
            if selected:
                _, p = self._ols(X[:, selected], y)
                worst = int(np.argmax(p[1:]))
                if p[1:][worst] > self.p_remove:
                    remaining.append(selected.pop(worst)); changed = True
            if not changed:
                break
        if not selected:                      # fall back to intercept + best single feature
            selected = [int(np.argmax([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]))]
        self.selected_ = selected
        self.coef_, _ = self._ols(X[:, selected], y)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.coef_[0] + X[:, self.selected_] @ self.coef_[1:]


if __name__ == "__main__":
    run_tabular_algorithm(
        "Stepwise_MLR", scaled_model(StepwiseLinearRegression(), scale_y=False),
        {"p_enter": [0.05, 0.10]},
        "Bidirectional stepwise multiple linear regression (p-value based entry/removal).")
