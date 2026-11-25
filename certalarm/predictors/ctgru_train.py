"""Training and prediction for CT-GRU.

A fresh observation mask is drawn for every sampled asset at every iteration, so the model learns
to tolerate the deployment's gaps instead of memorising one realisation. The returned model
carries an EMA of the weights: without it the validation curves swung by about 30% between
neighbouring checkpoints, which made any budget choice meaningless.

Iteration budgets were selected earlier on validation bearings and are fixed.
Never re-select them on calibration or test bearings.
"""
import copy

import numpy as np
import torch

from .. import observe
from .ctgru import CTGRU

ITERS = {"pronostia": 150, "xjtu": 750}
DEVICE = torch.device("cpu")


def _collate(items, device):
    """items: list of (X, rul, M). Right-pads to the longest asset; V marks the real steps."""
    B = len(items)
    L = max(len(r) for _, r, _ in items)
    Fd = items[0][0].shape[1]
    X = np.zeros((B, L, Fd), np.float32)
    M = np.zeros((B, L), np.float32)
    y = np.zeros((B, L), np.float32)
    V = np.zeros((B, L), np.float32)
    for i, (x, r, m) in enumerate(items):
        n = len(r)
        X[i, :n], y[i, :n], M[i, :n], V[i, :n] = x[:n], r, m[:n], 1.0
    t = lambda a: torch.as_tensor(a, device=device)
    return t(X), t(M), t(y), t(V)


LATE_TAU = 0.1        # the scale of "late life" for the alarm weighting, in normalized RUL


def _loss(mu, sigma, y, V, late_weight=0.0, asym=1.0, gate_asym=False):
    """Squared error plus a Gaussian NLL term for sigma.

    With the defaults this is the original loss exactly. The two knobs exist because of what the
    PRONOSTIA diagnosis found: both predictors over-estimate remaining life near failure (bias
    +0.21 and +0.11 in the last 20% of life), and CRC's threshold is set by the single most
    optimistic calibration bearing. Uniform symmetric MSE causes exactly that - most of the
    gradient comes from the long healthy phase the alarm never looks at, and reading high near
    failure costs no more than reading low.

    late_weight: extra weight on minutes near failure, 1 + late_weight * exp(-y / LATE_TAU).
    asym:        multiplier on over-estimation (mu > y), the direction that makes an alarm late.
    gate_asym:   confine the asymmetry to late life as well.

    `gate_asym` exists because the first search got this wrong. Applying the asymmetry at every
    minute made the predictor globally pessimistic - `strong` under-read remaining life by 0.41
    in early life - and since the alarm fires at the first crossing of a running minimum, a
    predictor biased low everywhere crosses the threshold far too early. That is waste, and it
    cancelled most of the gain from removing the late-life optimism. Gated, the asymmetry acts
    only where the alarm actually lives.
    """
    err = mu - y
    near = torch.exp(-y / LATE_TAU)
    w = 1.0 + late_weight * near
    a_scale = 1.0 + (asym - 1.0) * (near if gate_asym else torch.ones_like(near))
    a = torch.where(err > 0, a_scale, torch.ones_like(err))
    per = w * a * err ** 2 + 0.1 * (0.5 * ((y - mu.detach()) / sigma) ** 2 + torch.log(sigma))
    return ((per * V).sum(1) / V.sum(1).clamp_min(1e-6)).mean()


def train(X_list, rul_list, process, iters, seed, H=32, lr=1e-3, batch=32, ema_decay=0.98,
          device=DEVICE, late_weight=0.0, asym=1.0, gate_asym=False):
    """Train on the given assets under `process`. Returns the EMA model in eval mode."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    proc = observe.PROCESSES[process]
    model = CTGRU(X_list[0].shape[1], H=H).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = copy.deepcopy(model).eval()
    for _ in range(iters):
        idx = rng.integers(len(X_list), size=min(batch, len(X_list) * 8))
        items = [(X_list[i], rul_list[i], proc(rul_list[i], rng)["M"]) for i in idx]
        X, M, y, V = _collate(items, device)
        mu, sigma = model(X, M)
        loss = _loss(mu, sigma, y, V, late_weight, asym, gate_asym)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)
    ema.eval()
    return ema


@torch.no_grad()
def predict(model, X, M, device=DEVICE):
    """One asset -> (mu (L,), sigma (L,)) as float64 numpy arrays."""
    model.eval()
    xt = torch.as_tensor(np.asarray(X, np.float32)[None], device=device)
    mt = torch.as_tensor(np.asarray(M, np.float32)[None], device=device)
    mu, sigma = model(xt, mt)
    return mu[0].cpu().numpy().astype(np.float64), sigma[0].cpu().numpy().astype(np.float64)
