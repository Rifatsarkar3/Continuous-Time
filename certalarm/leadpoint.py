"""LeadPoint: a lightweight certified alarm built directly on the lead-point reduction.

The reduction (experiments/dev_lead_point_reduction.py, verified exactly) says that for an alarm
firing at the first crossing of a running minimum, an asset is late at threshold theta iff
theta <= theta*, where theta* is the running minimum of the signal at the last acceptable step.
Two consequences are used here.

  Certification needs one scalar per asset. `exact_threshold` is the split-conformal order
  statistic over {theta*_i}: no grid, no loss matrix, O(n log n), and never above the grid CRC
  threshold it replaces.

  Training can target the certificate directly. Certified waste is low when every asset's
  lead-point value sits below the other assets' healthy-phase scores, so `rank_loss` penalises
  exactly those cross-asset pairs, weighted by the life that would be wasted. Regression on
  remaining life spends capacity on ordering healthy states by age, which the alarm never uses.

The scorers are deliberately tiny (a linear map or one hidden layer on minute-grid features), and
the features add envelope-spectrum band energies - the demodulation physics of rolling-element
faults - to the existing 18 time/frequency features.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FS

ENV_EDGES = np.geomspace(10.0, 5000.0, 9)       # Hz; eight log-spaced envelope-spectrum bands
ENV_NAMES = [f"env_rel{i}" for i in range(8)] + ["log_env_power"]


def envelope_band_features(w, fs=FS):
    """(N, samples) raw windows -> (N, 9): relative log band energies of the envelope spectrum,
    plus the log of its total power. Relative energies make the band shape scale-free, the way
    DSAFLite's max-normalised envelope spectrum is; the total keeps the level."""
    x = np.asarray(w, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    h = np.zeros(n)
    h[0] = 1
    h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    env = np.abs(np.fft.ifft(np.fft.fft(x, axis=1) * h, axis=1))
    env = env - env.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(env, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = np.stack([P[:, (f >= lo) & (f < hi)].sum(1)
                     for lo, hi in zip(ENV_EDGES[:-1], ENV_EDGES[1:])], axis=1)
    tot = band.sum(axis=1, keepdims=True) + 1e-12
    return np.concatenate([np.log(band / tot + 1e-12), np.log(tot)], axis=1).astype(np.float32)


def minute_env_features(windows, per_min, chunk=512):
    """(n_win, samples) raw windows -> (n_min, 9); a partial final minute is dropped."""
    feats = np.concatenate([envelope_band_features(windows[i:i + chunk])
                            for i in range(0, len(windows), chunk)])
    n = len(feats) // per_min
    return feats[:n * per_min].reshape(n, per_min, -1).mean(1).astype(np.float32)


def env_grid(dataset, name, windows, per_min, cache_dir):
    """Cached envelope-band minute grid for one asset."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}_{name}.npy"
    if path.exists():
        return np.load(path)
    grid = minute_env_features(windows, per_min)
    np.save(path, grid)
    return grid


def ewma(X, halflife):
    """Causal exponentially weighted mean along axis 0; the value at t uses rows 0..t only."""
    X = np.asarray(X, dtype=np.float32)
    a = 1.0 - 2.0 ** (-1.0 / halflife)
    out = np.empty_like(X)
    acc = X[0].copy()
    for t in range(len(X)):
        acc = X[t] if t == 0 else (1.0 - a) * acc + a * X[t]
        out[t] = acc
    return out


def _last_ok(rul, lead):
    ok = np.flatnonzero(np.asarray(rul) >= lead)
    return int(ok[-1]) if ok.size else -1


def lead_point(signal, rul, lead):
    """theta*: the running minimum of the signal at the last step with remaining life >= lead."""
    t = _last_ok(rul, lead)
    return -math.inf if t < 0 else float(np.min(np.asarray(signal, float)[:t + 1]))


def exact_threshold(tstars, alpha):
    """Smallest theta with (#{theta*_i >= theta} + 1) / (n + 1) <= alpha, exactly.

    That is just above the (k+1)-th largest theta*, k = floor(alpha (n+1) - 1). Returns None below
    the certifiability floor (k < 0), where no threshold can be certified.
    """
    s = np.sort(np.asarray(tstars, float))[::-1]
    n = len(s)
    k = math.floor(round(alpha * (n + 1) - 1, 12))
    if k < 0:
        return None
    if k >= n:
        return -math.inf
    return float(np.nextafter(s[k], math.inf))


def fit_ridge(Xs, ruls, lam=1.0):
    """Linear ridge regression of normalized remaining life; the intercept is not penalised."""
    X = np.concatenate(Xs).astype(np.float64)
    y = np.concatenate(ruls).astype(np.float64)
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return {"kind": "ridge", "w": w, "b": float(ym - xm @ w)}


class RidgeModule(nn.Module):
    """A fitted ridge held as float64 torch state, so the write-once registry can hash it.

    `scale` (the fold-local feature scale) and `resid_sd` (the fit-set residual spread, used only
    by the uncertified baselines that need a predictive scale) ride along as buffers, so any
    change to them changes the registered hash too.
    """

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1, dtype=torch.float64)
        self.register_buffer("scale", torch.ones(d, dtype=torch.float32))
        self.register_buffer("resid_sd", torch.ones((), dtype=torch.float64))


def ridge_module(m, scale, resid_sd):
    mod = RidgeModule(len(m["w"]))
    with torch.no_grad():
        mod.lin.weight.copy_(torch.as_tensor(np.asarray(m["w"], np.float64)[None, :]))
        mod.lin.bias.fill_(float(m["b"]))
        mod.scale.copy_(torch.as_tensor(np.asarray(scale, np.float32)))
        mod.resid_sd.fill_(float(resid_sd))
    return mod.eval()


def module_score(mod, X):
    with torch.no_grad():
        return mod.lin(torch.as_tensor(np.asarray(X, np.float64))).squeeze(-1).numpy()


class _Scorer(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = (nn.Linear(d, 1) if hidden == 0 else
                    nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rank_loss(scores, ruls, lead):
    """Cross-asset lead-point ranking loss.

    For every pair (i, j) of training assets, each state of j that is still well before its own
    lead point should score above theta*_i; otherwise a threshold covering asset i fires on j at
    that state and wastes rul_j(t) - lead of its life. The softplus penalty on each pair is
    weighted by that waste and averaged over i and j (i = j included, which is the within-asset
    ordering).
    """
    th = torch.stack([s[:_last_ok(r, lead) + 1].min() for s, r in zip(scores, ruls)])
    per_j = []
    for s, r in zip(scores, ruls):
        v = torch.as_tensor(np.clip(np.asarray(r, np.float64) - lead, 0.0, None), dtype=s.dtype)
        per_j.append((F.softplus(th[:, None] - s[None, :]) * v).sum(1) / v.sum())
    return torch.stack(per_j).mean()


def fit_rank(Xs, ruls, lead, hidden=0, iters=300, lr=0.05, weight_decay=1e-2, seed=0):
    """Full-batch Adam on `rank_loss` plus an L2 penalty on the weights (not the biases)."""
    torch.manual_seed(seed)
    for r in ruls:
        if _last_ok(r, lead) < 0:
            raise ValueError("every training asset needs a step with remaining life >= lead")
    net = _Scorer(Xs[0].shape[1], hidden)
    X = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in Xs]
    weights = [p for n, p in net.named_parameters() if n.endswith("weight")]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = rank_loss([net(x) for x in X], ruls, lead) + \
            weight_decay * sum((p ** 2).sum() for p in weights)
        loss.backward()
        opt.step()
    net.eval()
    return {"kind": "rank", "net": net}


def score(model, X):
    """Alarm signal for one asset's (n_min, d) features; lower means closer to failure."""
    if model["kind"] == "ridge":
        return np.asarray(X, float) @ model["w"] + model["b"]
    with torch.no_grad():
        return model["net"](torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy().astype(float)


def n_params(model):
    if model["kind"] == "ridge":
        return int(model["w"].size + 1)
    return int(sum(p.numel() for p in model["net"].parameters()))
