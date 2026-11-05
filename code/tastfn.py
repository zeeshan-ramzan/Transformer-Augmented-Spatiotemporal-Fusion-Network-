"""
TASTFN - Transformer-Augmented Spatiotemporal Fusion Network (proposed method).

One execution of this file reproduces every TASTFN result of the paper:

  * main evaluation: LOYO-CV (2015-2024), independent test year (2025) and
    stratified 10-fold CV                                     (Tables 5, 6)
  * agreement, residual and error analyses                     (Tables 9, 10; Figs 3, 4)
  * training / inner-validation learning curves                (Fig. 5)
  * attention heat map and CMAG gate values by month           (Fig. 6)
  * ablation study                                             (Table 11)
  * permutation importance: standard, within-month, grouped    (Table 12, Fig. 7)
  * seasonal versus plot-level skill                           (Table 13)
  * seasonal performance                                       (Table 14)
  * NDVI-source sensitivity                                    (Table 15)
  * hyperparameter sensitivity                                 (Table 16)
  * model variants: log-target and forecast mode               (Table 17)
  * computational cost                                         (Table 18)

Run:
    python tastfn.py                      # full replication (GPU recommended)
    python tastfn.py --quick              # fast smoke test (meaningless numbers)
    python tastfn.py --skip-ablation --skip-hparam   # skip the most expensive parts

Architecture (paper, Methods): dual-stream input encoder (6 weather + 4 NDVI
features projected to d_model=64), phenological positional encoding (PPE),
2 pre-norm multi-head self-attention layers (8 heads, FFN 256) on the weather
stream, a two-layer feed-forward NDVI branch with masked mean pooling, the
cross-modal attention gating (CMAG) module, and a residual decoder [128, 64, 32].
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from common import (Reporter, between_plot_r2, cv_years, get_args, load_data,
                    loyo_folds, mean_sd, metrics, plt, standard_report,
                    stratified_folds, test_year)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_AGRO = len(C.AGRO_FEATURES)


# =============================================================================
# 1. Model
# =============================================================================
def positional_encoding(pos, month, d_model, kind):
    """pos, month: [B, T] tensors.  kind: 'ppe' | 'sin' | 'none'."""
    B, T = pos.shape
    if kind == "none":
        return torch.zeros(B, T, d_model, device=pos.device)
    i = torch.arange(0, d_model, 2, device=pos.device, dtype=torch.float32)
    div = torch.pow(10000.0, i / d_model)                       # 10000^(2i/d)
    ang = pos.unsqueeze(-1).float() / div                        # [B, T, d/2]
    pe = torch.zeros(B, T, d_model, device=pos.device)
    pe[..., 0::2] = torch.sin(ang)
    pe[..., 1::2] = torch.cos(ang)
    if kind == "ppe":                                            # Eqs. (3)-(4)
        pe = pe * torch.cos(2 * math.pi * month.float() / 12.0).unsqueeze(-1)
    return pe


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer: H + MHSA(LN(H)), then H + FFN(LN(H))."""

    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(p), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, pad_mask):
        z = self.ln1(x)
        a, w = self.attn(z, z, z, key_padding_mask=pad_mask, need_weights=True,
                         average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, w


class ResidualDecoder(nn.Module):
    """f(l+1) = ReLU(W f(l) + b) + P f(l)  (Eq. 11), widths 128-64-32."""

    def __init__(self, d_in, widths, p, residual=True):
        super().__init__()
        self.residual = residual
        self.lin, self.proj, w = nn.ModuleList(), nn.ModuleList(), d_in
        for u in widths:
            self.lin.append(nn.Linear(w, u))
            self.proj.append(nn.Linear(w, u, bias=False))
            w = u
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(w, 1)

    def forward(self, f):
        for lin, proj in zip(self.lin, self.proj):
            h = torch.relu(lin(f))
            f = self.drop(h + proj(f) if self.residual else h)
        return self.out(f).squeeze(-1)


class TASTFN(nn.Module):
    def __init__(self, n_agro=6, n_ndvi=4, d_model=64, n_heads=8, d_ff=256, n_layers=2,
                 decoder_widths=(128, 64, 32), dropout=0.30, attn_dropout=0.10,
                 pe="ppe", fusion="cmag", residual_decoder=True,
                 use_agro=True, use_ndvi=True, **_):
        super().__init__()
        self.d, self.pe_kind, self.fusion = d_model, pe, fusion
        self.use_agro, self.use_ndvi = use_agro, use_ndvi
        if use_agro:
            self.agro_proj = nn.Linear(n_agro, d_model)                 # Eq. (1)
            self.agro_ln = nn.LayerNorm(d_model)
            self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, attn_dropout)
                                         for _ in range(n_layers)])
            self.final_ln = nn.LayerNorm(d_model)
        if use_ndvi:
            self.ndvi_proj = nn.Linear(n_ndvi, d_model)                 # Eq. (2)
            self.ndvi_ln = nn.LayerNorm(d_model)
            self.ndvi_ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                         nn.Linear(d_model, d_model))
        if use_agro and use_ndvi:
            if fusion == "cmag":                                         # Eqs. (8)-(10)
                self.Wq = nn.Linear(d_model, d_model, bias=False)
                self.Wk = nn.Linear(d_model, d_model, bias=False)
                self.Wv = nn.Linear(d_model, d_model, bias=False)
                self.gate = nn.Linear(2 * d_model, d_model)
            else:                                                        # static concatenation
                self.concat = nn.Linear(2 * d_model, d_model)
        self.decoder = ResidualDecoder(d_model, decoder_widths, dropout, residual_decoder)

    @staticmethod
    def masked_mean(x, valid):
        w = valid.float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp(min=1.0)

    def forward(self, agro, ndvi, month, valid, return_attention=False):
        B, T = valid.shape
        pad = ~valid
        pos = torch.arange(T, device=valid.device).unsqueeze(0).expand(B, T)
        attn_w, alpha = None, None
        if self.use_agro:
            h = self.agro_ln(self.agro_proj(agro))
            h = h + positional_encoding(pos, month, self.d, self.pe_kind)
            for layer in self.layers:
                h, attn_w = layer(h, pad)
            Z = self.final_ln(h)
        if self.use_ndvi:
            hn = self.ndvi_ff(self.ndvi_ln(self.ndvi_proj(ndvi)))
            z_ndvi = self.masked_mean(hn, valid)
        if self.use_agro and self.use_ndvi:
            if self.fusion == "cmag":
                q = self.Wq(z_ndvi).unsqueeze(1)                          # [B,1,d]
                k, v = self.Wk(Z), self.Wv(Z)
                s = (q @ k.transpose(1, 2)) / math.sqrt(self.d)          # [B,1,T]
                s = s.masked_fill(pad.unsqueeze(1), float("-inf"))
                A = torch.softmax(s, dim=-1)
                c = (A @ v).squeeze(1)
                alpha = torch.sigmoid(self.gate(torch.cat([c, z_ndvi], -1)))
                f = alpha * c + (1 - alpha) * z_ndvi
            else:
                f = self.concat(torch.cat([self.masked_mean(Z, valid), z_ndvi], -1))
        elif self.use_agro:
            f = self.masked_mean(Z, valid)
        else:
            f = z_ndvi
        y = self.decoder(f)
        if return_attention:
            return y, attn_w, alpha
        return y


# =============================================================================
# 2. Sequence construction (Methods: temporal sequence construction)
# =============================================================================
class SequenceData:
    """Rolling windows of T consecutive monthly observations of one plot.

    Estimation mode (shift=0): window [t-T+1, ..., t] ends at the target month.
    Forecast mode  (shift=1): window [t-T, ..., t-1] ends one month earlier.
    """

    def __init__(self, df, context=6, shift=0, agro=C.AGRO_FEATURES, ndvi=C.NDVI_FEATURES):
        self.df, self.T, self.agro, self.ndvi = df, context, list(agro), list(ndvi)
        rows, targets = [], []
        for _, g in df.groupby(C.PLOT_COL, sort=False):
            idx = g.index.to_numpy()                 # already sorted by date
            for k in range(len(idx)):
                end = k - shift
                if end < 0:
                    continue                          # no context available
                win = [idx[j] if 0 <= j <= end else -1 for j in range(end - context + 1, end + 1)]
                rows.append(win)
                targets.append(idx[k])
        self.rows = np.array(rows, dtype=int)
        self.target = np.array(targets, dtype=int)
        self.year = df["year"].to_numpy()
        self.month = df["month"].to_numpy()
        self.y = df[C.TARGET_COL].to_numpy(float)
        self.t_year = self.year[self.target]
        self.t_month = self.month[self.target]

    def fit_scaler(self, train_years):
        m = self.df["year"].isin(train_years).to_numpy()
        X = self.df[self.agro + self.ndvi].to_numpy(float)
        self.mu, self.sd = X[m].mean(0), X[m].std(0) + 1e-9
        self.Xs = (X - self.mu) / self.sd

    def with_context(self, sel, exclude_year=None):
        """Boolean selection restricted to windows that keep >=1 valid position."""
        sel = np.asarray(sel, bool).copy()
        R = self.rows
        valid = R >= 0
        if exclude_year is not None:
            valid &= self.year[np.clip(R, 0, None)] != exclude_year
        return sel & valid.any(1)

    def tensors(self, sel, exclude_year=None):
        R = self.rows[sel]
        valid = R >= 0
        Rc = np.clip(R, 0, None)
        if exclude_year is not None:                  # no inputs from the held-out year
            valid &= self.year[Rc] != exclude_year
        F = self.Xs[Rc] * valid[..., None]
        na = len(self.agro)
        out = {"agro": F[..., :na], "ndvi": F[..., na:], "month": self.month[Rc] * valid,
               "valid": valid, "y": self.y[self.target[sel]]}
        return {k: torch.as_tensor(v, dtype=torch.bool if k == "valid" else
                                   (torch.long if k == "month" else torch.float32))
                for k, v in out.items()}

    def frame(self, sel, pred):
        t = self.target[sel]
        return pd.DataFrame({"plot_id": self.df[C.PLOT_COL].to_numpy()[t], "year": self.year[t],
                             "month": self.month[t], "observed": self.y[t],
                             "predicted": np.asarray(pred, float), "row": t})


# =============================================================================
# 3. Training (AdamW, cosine annealing with warm restarts, early stopping)
# =============================================================================
def predict(model, D, batch=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(D["y"]), batch):
            out.append(model(*(D[k][i:i + batch].to(DEVICE) for k in ("agro", "ndvi", "month", "valid"))).cpu())
    return torch.cat(out).numpy()


def train_tastfn(cfg, Dtr, Dva, seed=C.RANDOM_STATE, log_target=False):
    """Returns (model, history, best_epoch, y-transform)."""
    torch.manual_seed(seed); np.random.seed(seed)
    ytr = Dtr["y"].numpy()
    yt = np.log(ytr) if log_target else ytr
    ym, ysd = float(yt.mean()), float(yt.std() + 1e-9)

    def to_t(y):
        y = np.log(y) if log_target else y
        return torch.as_tensor((y - ym) / ysd, dtype=torch.float32)

    def inv(p):
        p = p * ysd + ym
        return np.exp(p) if log_target else p

    model = TASTFN(n_agro=Dtr["agro"].shape[-1], n_ndvi=Dtr["ndvi"].shape[-1], **cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg["t0"], eta_min=cfg["eta_min"])
    ytr_t = to_t(ytr)
    n = len(ytr_t)
    hist = {"train_rmse": [], "val_rmse": []}
    best, best_state, best_ep, wait = np.inf, None, 0, 0
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        se, cnt = 0.0, 0
        for i in range(0, n, cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            xb = [Dtr[k][b].to(DEVICE) for k in ("agro", "ndvi", "month", "valid")]
            yb = ytr_t[b].to(DEVICE)
            opt.zero_grad()
            pred = model(*xb)
            loss = torch.mean((pred - yb) ** 2)               # MSE loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                pr = inv(pred.detach().cpu().numpy())
                se += float(np.sum((pr - ytr[b.numpy()]) ** 2)); cnt += len(b)
        sched.step()
        hist["train_rmse"].append(math.sqrt(se / cnt))
        vr = metrics(Dva["y"].numpy(), inv(predict(model, Dva)))["RMSE"]
        hist["val_rmse"].append(vr)
        if np.isfinite(vr) and vr < best - 1e-6:
            best, best_ep, wait = vr, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"], hist["stop_epoch"], hist["best_val"] = best_ep, len(hist["val_rmse"]), best
    model.inv = inv
    return model, hist


# =============================================================================
# 4. Evaluation protocols
# =============================================================================
def run_loyo(df, cfg, S, log_target=False, keep_models=False, label=""):
    rows, preds, hists, models = [], [], [], []
    for outer, train_years, inner in loyo_folds(df):
        S.fit_scaler(train_years)
        tr = S.with_context(np.isin(S.t_year, [t for t in train_years if t != inner]), outer)
        va = S.with_context(S.t_year == inner, outer)
        te = S.t_year == outer
        if te.sum() == 0 or tr.sum() == 0 or va.sum() == 0:
            continue
        model, h = train_tastfn(cfg, S.tensors(tr, outer), S.tensors(va, outer), log_target=log_target)
        Dte = S.tensors(te)
        p = model.inv(predict(model, Dte))
        rows.append({"year": outer, **metrics(Dte["y"].numpy(), p)})
        preds.append(S.frame(te, p))
        hists.append(h)
        if keep_models:
            models.append((outer, model, te))
        print(f"  {label} LOYO fold {outer}: RMSE {rows[-1]['RMSE']:.2f} "
              f"(best epoch {h['best_epoch']}, stopped {h['stop_epoch']})", flush=True)
    return rows, pd.concat(preds, ignore_index=True), hists, models


def run_test(df, cfg, S, log_target=False):
    years, ty = cv_years(df), test_year(df)
    S.fit_scaler(years)
    tr = np.isin(S.t_year, years[:-1]); va = S.t_year == years[-1]; te = S.t_year == ty
    model, h = train_tastfn(cfg, S.tensors(tr), S.tensors(va), log_target=log_target)
    Dte = S.tensors(te)
    t0 = time.time(); p = model.inv(predict(model, Dte)); dt = time.time() - t0
    return S.frame(te, p), model, h, dt


def run_stratified(df, cfg, S, n_folds):
    """Random stratified K-fold on targets (overlapping windows allowed; optimistic)."""
    S.fit_scaler(sorted(df["year"].unique()))
    sub = df.loc[S.target]
    folds = stratified_folds(sub.reset_index(drop=True), n_splits=C.N_STRATIFIED_FOLDS)[:n_folds]
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows, preds = [], []
    for k, (a, b) in enumerate(folds):
        a = rng.permutation(a); nv = max(1, len(a) // 10)
        va, tr = np.sort(a[:nv]), np.sort(a[nv:])      # 10% of training fold for early stopping
        model, _ = train_tastfn(cfg, S.tensors(tr), S.tensors(va))
        Dte = S.tensors(b)
        p = model.inv(predict(model, Dte))
        rows.append({"fold": k + 1, **metrics(Dte["y"].numpy(), p)})
        preds.append(S.frame(b, p))
        print(f"  stratified fold {k + 1}: RMSE {rows[-1]['RMSE']:.2f}", flush=True)
    return rows, pd.concat(preds, ignore_index=True)


# =============================================================================
# 5. Interpretability
# =============================================================================
def attention_and_gates(models, S, df):
    """Mean layer-2 attention (all heads, held-out windows with full context)
    and mean CMAG gate value by calendar month."""
    att_sum, att_n, gate_rows = None, 0, []
    for outer, model, te in models:
        S.fit_scaler([y for y in cv_years(df) if y != outer])
        D = S.tensors(te)
        model.eval()
        with torch.no_grad():
            _, w, alpha = model(*(D[k].to(DEVICE) for k in ("agro", "ndvi", "month", "valid")),
                                return_attention=True)
        full = D["valid"].all(1).numpy()
        if w is not None and full.any():
            ww = w.cpu().numpy()[full]
            att_sum = ww.sum(0) if att_sum is None else att_sum + ww.sum(0)
            att_n += len(ww)
        if alpha is not None:
            a = alpha.mean(1).cpu().numpy()
            gate_rows.append(pd.DataFrame({"month": S.t_month[te], "alpha": a}))
    att = att_sum / att_n if att_n else None
    gates = pd.concat(gate_rows).groupby("month")["alpha"].mean() if gate_rows else None
    return att, gates


def permutation_importance(models, S, df, n_rep, rng):
    names = list(S.agro + S.ndvi)
    groups = {f: [names.index(f)] for f in names}
    groups["Temperature group (min, max, mean; grouped)"] = [names.index(f) for f in C.TEMPERATURE_GROUP]
    std_inc = {g: [] for g in groups}
    cond_inc = {g: [] for g in groups}
    for outer, model, te in models:
        S.fit_scaler([y for y in cv_years(df) if y != outer])
        D = S.tensors(te)
        y = D["y"].numpy()
        base = metrics(y, model.inv(predict(model, D)))["RMSE"]
        months = S.t_month[te]
        X = torch.cat([D["agro"], D["ndvi"]], -1)
        na = D["agro"].shape[-1]
        for g, cols in groups.items():
            for kind, store in (("std", std_inc), ("cond", cond_inc)):
                incs = []
                for _ in range(n_rep):
                    if kind == "std":
                        perm = rng.permutation(len(y))
                    else:                                   # shuffle within calendar month
                        perm = np.arange(len(y))
                        for m in np.unique(months):
                            ix = np.where(months == m)[0]
                            perm[ix] = rng.permutation(ix)
                    Xp = X.clone()
                    Xp[:, :, cols] = X[perm][:, :, cols]
                    Dp = dict(D, agro=Xp[..., :na], ndvi=Xp[..., na:])
                    incs.append(metrics(y, model.inv(predict(model, Dp)))["RMSE"] - base)
                store[g].append(float(np.mean(incs)))
    out = pd.DataFrame({"feature": [C.FEATURE_LABELS.get(g, g) for g in groups],
                        "standard": [np.mean(std_inc[g]) for g in groups],
                        "conditional_within_month": [np.mean(cond_inc[g]) for g in groups]})
    return out.sort_values("conditional_within_month", ascending=False).reset_index(drop=True)


# =============================================================================
# 6. Figures specific to TASTFN
# =============================================================================
def plot_learning_curves(rep, hists):
    L = max(len(h["train_rmse"]) for h in hists)
    tr = np.full((len(hists), L), np.nan); va = np.full((len(hists), L), np.nan)
    for i, h in enumerate(hists):
        tr[i, :len(h["train_rmse"])] = h["train_rmse"]; va[i, :len(h["val_rmse"])] = h["val_rmse"]
    ep = np.arange(1, L + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for arr, c, lab, ls in ((tr, "#3b7dc4", "Training RMSE", "-"), (va, "#d0453a", "Inner-validation RMSE", "--")):
        m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
        ax.plot(ep, m, ls, color=c, label=f"{lab} (mean ± SD, {len(hists)} folds)")
        ax.fill_between(ep, m - s, m + s, color=c, alpha=0.18)
    for x in range(C.TASTFN_PARAMS["t0"], L, C.TASTFN_PARAMS["t0"]):
        ax.axvline(x, color="#bbb", ls=":", lw=0.8)
    best = np.mean([h["best_epoch"] for h in hists]); stop = np.mean([h["stop_epoch"] for h in hists])
    ax.axvline(stop, color="#333", ls="--", lw=1.2, label=f"Mean early-stopping epoch ({stop:.0f})")
    vm = np.nanmean(va, 0)
    bi = int(np.nanargmin(vm))
    ax.plot(ep[bi], vm[bi], "o", mfc="white", mec="k", ms=8,
            label=f"Minimum mean validation RMSE {vm[bi]:.2f} (epoch {ep[bi]})")
    ax.set_yscale("log"); ax.set_xlabel("Training epoch"); ax.set_ylabel("RMSE (kg ha$^{-1}$)")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=7)
    rep.save_fig(fig, "fig_learning_curves.png")
    gap = float(np.nanmean(va[:, bi]) - np.nanmean(tr[:, bi]))
    return {"mean_best_epoch": best, "mean_stop_epoch": stop, "min_mean_val_rmse": float(vm[bi]),
            "epoch_of_min": int(ep[bi]), "train_val_gap_at_min": gap}


def plot_attention(rep, att):
    T = att.shape[0]
    lab = [f"t-{T - 1 - i}" if i < T - 1 else "t" for i in range(T)]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(att, cmap="viridis", vmin=0)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{att[i, j]:.2f}", ha="center", va="center",
                    color="white" if att[i, j] < att.max() * 0.6 else "black", fontsize=8)
    ax.set_xticks(range(T), lab); ax.set_yticks(range(T), lab)
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    ax.set_title("Mean self-attention, encoder layer 2 (held-out windows)", fontsize=9)
    fig.colorbar(im, ax=ax, label="Attention weight")
    rep.save_fig(fig, "fig_attention_heatmap.png")


def plot_gates(rep, gates):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = gates.index.to_numpy()
    ax.plot(x, gates.values, "o-", color="#3b7dc4", label=r"Cross-modal context weight $\alpha$")
    ax.plot(x, 1 - gates.values, "s--", color="#d0453a", label=r"NDVI weight $1-\alpha$")
    ax.set_xticks(x, [names[m - 1] for m in x]); ax.set_ylim(0, 1)
    ax.set_ylabel("Mean gate value"); ax.legend(fontsize=8)
    rep.save_fig(fig, "fig_gate_by_month.png")


def plot_importance(rep, imp):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yy = np.arange(len(imp))[::-1]
    ax.barh(yy + 0.2, imp["standard"], 0.4, color="#8fb3d9", ec="k", lw=0.4, label="Standard permutation")
    ax.barh(yy - 0.2, imp["conditional_within_month"], 0.4, color="#2a8c74", ec="k", lw=0.4,
            label="Conditional (within-month) permutation")
    ax.set_yticks(yy, imp["feature"]); ax.set_xlabel("Mean increase in RMSE (kg ha$^{-1}$)")
    ax.legend(fontsize=8)
    rep.save_fig(fig, "fig_permutation_importance.png")


def plot_bar(rep, df, x, y, fname, ylabel):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(df[x], df[y], color="#3b7dc4", ec="k", lw=0.4)
    ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    rep.save_fig(fig, fname)


# =============================================================================
# 7. Main
# =============================================================================
def main():
    def extra(p):
        p.add_argument("--skip-ablation", action="store_true")
        p.add_argument("--skip-importance", action="store_true")
        p.add_argument("--skip-sensitivity", action="store_true", help="NDVI-source sensitivity")
        p.add_argument("--skip-variants", action="store_true", help="log-target / forecast mode")
        p.add_argument("--skip-hparam", action="store_true")
        p.add_argument("--hp-protocol", choices=["inner", "stratified"], default="inner",
                       help="hyperparameter sensitivity on the inner validation year (default) "
                            "or with stratified 10-fold CV")
        p.add_argument("--n-perm", type=int, default=50)

    args = get_args("TASTFN - proposed method, full replication", extra)
    df = load_data(args.data)
    cfg = dict(C.TASTFN_PARAMS)
    n_strat, n_perm = C.N_STRATIFIED_FOLDS, args.n_perm
    if args.quick:
        cfg.update(max_epochs=3, patience=2)
        n_strat, n_perm = 2, 2
    rng = np.random.default_rng(C.RANDOM_STATE)
    rep = Reporter(args.out, "TASTFN")
    rep.text("TASTFN - Transformer-Augmented Spatiotemporal Fusion Network")
    rep.text(f"Device: {DEVICE}   Data: {args.data} ({len(df)} observations, "
             f"{df[C.PLOT_COL].nunique()} plots, years {df.year.min()}-{df.year.max()})")
    rep.text(f"Configuration: {cfg}")
    rep.text(f"Test year: {test_year(df)}   LOYO-CV years: {cv_years(df)}")

    S = SequenceData(df, cfg["context"], 0)

    # ---------------- main evaluation ----------------
    t0 = time.time()
    loyo_rows, loyo_pred, hists, models = run_loyo(df, cfg, S, keep_models=True, label="TASTFN")
    loyo_time = time.time() - t0
    test_pred, test_model, test_hist, infer_t = run_test(df, cfg, S)
    strat_rows, strat_pred = run_stratified(df, cfg, S, n_strat)
    loyo_m, pooled_m, test_m, strat_m = standard_report(rep, df, loyo_rows, loyo_pred, test_pred,
                                                        strat_rows, strat_pred)

    rep.section("Learning curves (Fig. 5)")
    lc = plot_learning_curves(rep, hists)
    rep.text(f"Minimum mean inner-validation RMSE {lc['min_mean_val_rmse']:.2f} kg/ha at epoch "
             f"{lc['epoch_of_min']}; mean best epoch {lc['mean_best_epoch']:.0f}; mean early-stopping "
             f"epoch {lc['mean_stop_epoch']:.0f}; train/validation gap {lc['train_val_gap_at_min']:.2f} kg/ha")
    rep.json["learning_curves"] = lc

    rep.section("Attention and CMAG gate values (Fig. 6)")
    att, gates = attention_and_gates(models, S, df)
    if att is not None:
        plot_attention(rep, att)
        rep.table(pd.DataFrame(att, columns=[f"key_{i}" for i in range(att.shape[1])]),
                  "table_attention_matrix.csv")
    if gates is not None:
        plot_gates(rep, gates)
        rep.table(gates.reset_index().rename(columns={"alpha": "mean_alpha"}), "table_gate_by_month.csv")

    # ---------------- ablation ----------------
    if not args.skip_ablation:
        rep.section("Ablation study under LOYO-CV (Table 11)")
        steps = [("Baseline MLP (3 layers)", dict(context=1, n_layers=0, pe="none", fusion="concat", residual_decoder=False)),
                 ("+ Temporal sequencing", dict(n_layers=0, pe="none", fusion="concat", residual_decoder=False)),
                 ("+ MHSA encoder (L=1)", dict(n_layers=1, pe="sin", fusion="concat", residual_decoder=False)),
                 ("+ MHSA encoder (L=2)", dict(n_layers=2, pe="sin", fusion="concat", residual_decoder=False)),
                 ("+ Phenological PE", dict(n_layers=2, pe="ppe", fusion="concat", residual_decoder=False)),
                 ("+ CMAG module", dict(n_layers=2, pe="ppe", fusion="cmag", residual_decoder=False)),
                 ("+ Residual decoder (full TASTFN)", None)]
        abl = []
        for name, change in steps:
            if change is None:
                m = loyo_m
            else:
                c2 = dict(cfg, **change)
                S2 = SequenceData(df, c2["context"], 0)
                r, _, _, _ = run_loyo(df, c2, S2, label=name)
                m = mean_sd(r)
            abl.append({"configuration": name, "R2": m["R2"], "R2_sd": m["R2_sd"], "RMSE": m["RMSE"],
                        "RMSE_sd": m["RMSE_sd"], "MAE": m["MAE"], "MAPE": m["MAPE"]})
        abl = pd.DataFrame(abl)
        abl["RMSE_reduction_%"] = np.r_[np.nan, 100 * (1 - abl["RMSE"].values[1:] / abl["RMSE"].values[:-1])]
        rep.table(abl, "table_ablation.csv")
        plot_bar(rep, abl, "configuration", "RMSE", "fig_ablation_rmse.png", "LOYO-CV RMSE (kg ha$^{-1}$)")
        rep.json["ablation"] = abl.to_dict("records")

    # ---------------- permutation importance & plot-level skill ----------------
    if not args.skip_importance:
        rep.section(f"Permutation importance under LOYO-CV, {n_perm} permutations (Table 12, Fig. 7)")
        imp = permutation_importance(models, S, df, n_perm, rng)
        rep.table(imp, "table_permutation_importance.csv")
        plot_importance(rep, imp)
        rep.json["importance"] = imp.to_dict("records")

        rep.section("Seasonal versus plot-level skill under LOYO-CV (Table 13)")
        rows = []
        clim = []
        for outer, train_years, _ in loyo_folds(df):
            mm = df[df.year.isin(train_years)].groupby("month")[C.TARGET_COL].mean()
            d = df[df.year == outer]
            clim.append(pd.DataFrame({"plot_id": d[C.PLOT_COL], "year": d.year, "month": d.month,
                                      "observed": d[C.TARGET_COL],
                                      "predicted": d.month.map(mm).fillna(mm.mean())}))
        clim = pd.concat(clim)
        mc = metrics(clim.observed, clim.predicted)
        rows.append({"model": "Monthly climatology", "R2": mc["R2"], "RMSE": mc["RMSE"], "between_plot_R2": 0.0})
        for name, change in (("TASTFN, weather stream only", dict(use_ndvi=False)),
                             ("TASTFN, NDVI stream only", dict(use_agro=False))):
            r, p, _, _ = run_loyo(df, dict(cfg, **change), S, label=name)
            m = mean_sd(r)
            rows.append({"model": name, "R2": m["R2"], "RMSE": m["RMSE"], "between_plot_R2": between_plot_r2(p)})
        rows.append({"model": "TASTFN, full model", "R2": loyo_m["R2"], "RMSE": loyo_m["RMSE"],
                     "between_plot_R2": between_plot_r2(loyo_pred)})
        pl = pd.DataFrame(rows)
        rep.table(pl, "table_plot_level_skill.csv")
        rep.json["plot_level"] = pl.to_dict("records")

    # ---------------- NDVI-source sensitivity ----------------
    if not args.skip_sensitivity:
        rep.section("NDVI-source sensitivity under LOYO-CV (Table 15)")
        rows = [{"configuration": "Landsat-8 OLI (primary)", **{k: loyo_m[k] for k in ("R2", "RMSE", "MAPE")}}]
        configs = [("NDVI mean only", df, ["ndvi_mean"])]
        if all(c in df.columns for c in C.L78_NDVI_FEATURES):
            d = df.copy()
            d[C.NDVI_FEATURES] = d[C.L78_NDVI_FEATURES].values
            configs.insert(0, ("Landsat-7/8 combined", d, C.NDVI_FEATURES))
        d17 = df[df.year >= 2017].reset_index(drop=True)
        d17["row_id"] = np.arange(len(d17))
        configs.append(("Landsat-8 OLI, 2017 onwards", d17, C.NDVI_FEATURES))
        if all(c in df.columns for c in C.S2_NDVI_FEATURES):
            ds = d17.dropna(subset=C.S2_NDVI_FEATURES).copy()
            ds[C.NDVI_FEATURES] = ds[C.S2_NDVI_FEATURES].values
            configs.append(("Sentinel-2 MSI (10 m), 2017 onwards", ds.reset_index(drop=True), C.NDVI_FEATURES))
        for name, d, nd in configs:
            if d["year"].nunique() < 3:
                continue
            S2 = SequenceData(d, cfg["context"], 0, ndvi=nd)
            r, _, _, _ = run_loyo(d, cfg, S2, label=name)
            m = mean_sd(r)
            rows.append({"configuration": name, **{k: m[k] for k in ("R2", "RMSE", "MAPE")}})
        if C.GAPFILL_COL in df.columns:
            ok = loyo_pred["row"].map(df[C.GAPFILL_COL]).fillna(1).values == 0
            if ok.sum() > 1:
                m = metrics(loyo_pred.observed[ok], loyo_pred.predicted[ok])
                rows.append({"configuration": f"Primary model, observations without NDVI gap-filling (n={ok.sum()})",
                             **{k: m[k] for k in ("R2", "RMSE", "MAPE")}})
        else:
            rep.text(f"(column '{C.GAPFILL_COL}' not found: gap-filling analysis skipped)")
        sens = pd.DataFrame(rows)
        rep.table(sens, "table_ndvi_sensitivity.csv")
        rep.json["ndvi_sensitivity"] = sens.to_dict("records")

    # ---------------- variants ----------------
    if not args.skip_variants:
        rep.section("Model variants: log-transformed target and forecast mode (Table 17)")
        rows = [{"variant": "Estimation mode, original-scale target (primary)", "R2": loyo_m["R2"],
                 "RMSE": loyo_m["RMSE"], "MAE": loyo_m["MAE"], "MAPE": loyo_m["MAPE"], "test_RMSE": test_m["RMSE"]}]
        r, p, _, _ = run_loyo(df, cfg, S, log_target=True, label="log-target")
        tp, _, _, _ = run_test(df, cfg, S, log_target=True)
        m = mean_sd(r)
        rows.append({"variant": "Estimation mode, log-transformed target", **{k: m[k] for k in ("R2", "RMSE", "MAE", "MAPE")},
                     "test_RMSE": metrics(tp.observed, tp.predicted)["RMSE"]})
        Sf = SequenceData(df, cfg["context"], 1)
        r, p, _, _ = run_loyo(df, cfg, Sf, label="forecast")
        tp, _, _, _ = run_test(df, cfg, Sf)
        m = mean_sd(r)
        rows.append({"variant": "Forecast mode (one-month lead)", **{k: m[k] for k in ("R2", "RMSE", "MAE", "MAPE")},
                     "test_RMSE": metrics(tp.observed, tp.predicted)["RMSE"]})
        var = pd.DataFrame(rows)
        rep.table(var, "table_model_variants.csv")
        rep.json["variants"] = var.to_dict("records")

    # ---------------- hyperparameter sensitivity ----------------
    if not args.skip_hparam:
        rep.section(f"Hyperparameter sensitivity ({args.hp_protocol} protocol) (Table 16)")
        grid = [("context", [3, 4, 5, 6, 8, 12]), ("n_heads", [1, 2, 4, 8, 16]),
                ("n_layers", [1, 2, 3, 4]), ("d_model", [32, 64, 128, 256])]
        years = cv_years(df)
        rows = []
        for hp, values in grid:
            for v in values:
                c2 = dict(cfg, **{hp: v})
                if c2["d_model"] % c2["n_heads"]:
                    continue
                S2 = SequenceData(df, c2["context"], 0)
                if args.hp_protocol == "inner":
                    S2.fit_scaler(years[:-1])
                    tr = np.isin(S2.t_year, years[:-2]); va = S2.t_year == years[-2]; te = S2.t_year == years[-1]
                    mdl, _ = train_tastfn(c2, S2.tensors(tr), S2.tensors(va))
                    Dte = S2.tensors(te)
                    rmse = metrics(Dte["y"].numpy(), mdl.inv(predict(mdl, Dte)))["RMSE"]
                else:
                    r, _ = run_stratified(df, c2, S2, n_strat)
                    rmse = mean_sd(r)["RMSE"]
                sel = " (selected)" if v == cfg[hp] else ""
                rows.append({"hyperparameter": hp, "value": f"{v}{sel}", "RMSE": rmse})
                print(f"  {hp}={v}: RMSE {rmse:.2f}", flush=True)
        hpd = pd.DataFrame(rows)
        if args.hp_protocol == "inner":
            rep.text(f"Trained on {years[0]}-{years[-3]}, early stopping on {years[-2]}, RMSE on {years[-1]}")
        rep.table(hpd, "table_hyperparameter_sensitivity.csv")
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, (hp, _) in zip(axes, grid):
            d = hpd[hpd.hyperparameter == hp]
            ax.plot(range(len(d)), d["RMSE"], "o-", color="#3b7dc4")
            ax.set_xticks(range(len(d)), [s.replace(" (selected)", "*") for s in d["value"]])
            ax.set_title(hp, fontsize=9); ax.set_ylabel("RMSE (kg ha$^{-1}$)")
        rep.save_fig(fig, "fig_hyperparameter_sensitivity.png")
        rep.json["hyperparameter_sensitivity"] = hpd.to_dict("records")

    # ---------------- computational cost ----------------
    rep.section("Computational cost (Table 18)")
    n_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    flops = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        S.fit_scaler(cv_years(df))
        D1 = S.tensors(np.array([0]))
        with FlopCounterMode(display=False) as fc:
            test_model(*(D1[k].to(DEVICE) for k in ("agro", "ndvi", "month", "valid")))
        flops = fc.get_total_flops()
    except Exception as e:  # pragma: no cover
        rep.text(f"(FLOP counting unavailable: {e})")
    ops = len(test_pred) / infer_t if infer_t > 0 else float("nan")
    cost = pd.DataFrame([{"parameters": n_params, "FLOPs_per_observation": flops,
                          "LOYO_CV_train_time_s": loyo_time, "inference_obs_per_s": ops}])
    rep.table(cost, "table_computational_cost.csv", floatfmt="{:,.1f}")
    rep.json["cost"] = cost.to_dict("records")[0]

    torch.save(test_model.state_dict(), rep.dir / "tastfn_final_model.pt")
    rep.text("Final model (trained on all CV years) saved to tastfn_final_model.pt")
    rep.finish()


if __name__ == "__main__":
    main()
