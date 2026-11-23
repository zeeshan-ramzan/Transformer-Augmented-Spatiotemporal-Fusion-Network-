"""Gaussian process regression, Matern 5/2 kernel.  Run:  python gpr_matern52.py"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import config as C
from common import run_tabular_algorithm, scaled_model

if __name__ == "__main__":
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=1,
                                   random_state=C.RANDOM_STATE)
    # Kernel hyperparameters are fitted by maximizing the marginal likelihood on the
    # training years; the number of optimizer restarts is selected on the inner year.
    run_tabular_algorithm(
        "GPR_matern52", scaled_model(gpr), {"n_restarts_optimizer": [0, 1]},
        "Gaussian process regression with a Matern 5/2 kernel plus white noise.",
        quick_grid={"n_restarts_optimizer": [0]})
