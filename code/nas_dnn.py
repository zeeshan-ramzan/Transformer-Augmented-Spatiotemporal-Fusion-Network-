"""
NAS-DNN baseline (Ramzan et al., IEEE Access 2023): a fully connected deep
regression network whose architecture is found by neural architecture search
with Bayesian optimization.  Run:  python nas_dnn.py

Search space: 1-3 hidden layers, 8-128 units per layer, ReLU/tanh/sigmoid
activation, learning rate 1e-4..1e-2 (log), L2 weight decay 1e-6..1e-2 (log).
Search strategy: Bayesian optimization with a Gaussian-process surrogate and
expected improvement, evaluated on the inner validation year of each fold.
The selected architecture is then refitted on all training years for the
number of epochs at which the inner-validation RMSE was lowest.
"""
import numpy as np
import torch
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

import config as C
from common import metrics, run_tabular_algorithm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACT = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}


class TorchDNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_layers=3, units1=64, units2=32, units3=16, activation="relu",
                 lr=1e-3, weight_decay=1e-4, epochs=300, batch_size=32, patience=30, seed=C.RANDOM_STATE):
        self.n_layers, self.units1, self.units2, self.units3 = n_layers, units1, units2, units3
        self.activation, self.lr, self.weight_decay = activation, lr, weight_decay
        self.epochs, self.batch_size, self.patience, self.seed = epochs, batch_size, patience, seed

    def _net(self, d):
        layers, w = [], d
        for u in [self.units1, self.units2, self.units3][: self.n_layers]:
            layers += [torch.nn.Linear(w, int(u)), ACT[self.activation]()]
            w = int(u)
        layers.append(torch.nn.Linear(w, 1))
        return torch.nn.Sequential(*layers).to(DEVICE)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.xs_ = StandardScaler().fit(X)
        self.ym_, self.ysd_ = float(np.mean(y)), float(np.std(y) + 1e-9)
        Xt = torch.tensor(self.xs_.transform(X), dtype=torch.float32, device=DEVICE)
        yt = torch.tensor((y - self.ym_) / self.ysd_, dtype=torch.float32, device=DEVICE)
        self.net_ = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait, self.best_epoch_ = np.inf, None, 0, self.epochs
        for ep in range(int(self.epochs)):
            self.net_.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = torch.mean((self.net_(Xt[b]).squeeze(-1) - yt[b]) ** 2)
                loss.backward(); opt.step()
            if X_val is not None:
                r = metrics(y_val, self.predict(X_val))["RMSE"]
                if r < best - 1e-6:
                    best, wait, self.best_epoch_ = r, 0, ep + 1
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.val_rmse_ = best
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.xs_.transform(X), dtype=torch.float32, device=DEVICE)
            return self.net_(Xt).squeeze(-1).cpu().numpy() * self.ysd_ + self.ym_


# ---------------------------------------------------------------------------
# Bayesian optimization over the architecture search space
# ---------------------------------------------------------------------------
ACTS = ["relu", "tanh", "sigmoid"]


def _sample(rng):
    return {"n_layers": int(rng.integers(1, 4)), "units1": int(rng.integers(8, 129)),
            "units2": int(rng.integers(8, 129)), "units3": int(rng.integers(8, 129)),
            "activation": ACTS[int(rng.integers(0, 3))],
            "lr": float(10 ** rng.uniform(-4, -2)), "weight_decay": float(10 ** rng.uniform(-6, -2))}


def _encode(c):
    return [c["n_layers"] / 3, c["units1"] / 128, c["units2"] / 128, c["units3"] / 128,
            *[float(c["activation"] == a) for a in ACTS],
            (np.log10(c["lr"]) + 4) / 2, (np.log10(c["weight_decay"]) + 6) / 4]


def bayes_nas(model, Xtr, ytr, Xva, yva, quick):
    rng = np.random.default_rng(C.RANDOM_STATE)
    n_init, n_iter, max_ep = (2, 1, 20) if quick else (6, 14, 400)
    pool = [_sample(rng) for _ in range(300)]
    tried, scores, epochs = [], [], []

    def evaluate(cfg):
        m = TorchDNNRegressor(**cfg, epochs=max_ep).fit(Xtr, ytr, Xva, yva)
        tried.append(cfg); scores.append(m.val_rmse_); epochs.append(m.best_epoch_)

    for cfg in pool[:n_init]:
        evaluate(cfg)
    cand = pool[n_init:]
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(1e-3), normalize_y=True)
        gp.fit(np.array([_encode(c) for c in tried]), np.array(scores))
        mu, sd = gp.predict(np.array([_encode(c) for c in cand]), return_std=True)
        imp = min(scores) - mu
        z = imp / np.maximum(sd, 1e-9)
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)
        evaluate(cand.pop(int(np.argmax(ei))))
    k = int(np.argmin(scores))
    return {**tried[k], "epochs": int(epochs[k])}


if __name__ == "__main__":
    run_tabular_algorithm(
        "NAS_DNN", TorchDNNRegressor(), {},
        "Deep regression network designed by neural architecture search (Bayesian optimization).",
        tuner=bayes_nas)
