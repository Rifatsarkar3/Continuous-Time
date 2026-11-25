"""Observation processes on the minute grid.

Each process returns, for one bearing, dict(M, link): M marks minutes carrying an acquired
measurement, link marks minutes when the telemetry link was up. Nothing is acquired while the
link is down, and the commissioning period is always observed because the feature baseline needs
it. Unchanged from the earlier pre-registered runs, so that results remain comparable.
"""
import numpy as np

from .features import BASELINE_MIN  # the commissioning window the feature baseline needs


def _finish(M, link):
    M = np.asarray(M, dtype=np.float32)
    link = np.asarray(link, dtype=np.float32)
    M[:BASELINE_MIN] = 1.0
    link[:BASELINE_MIN] = 1.0
    return {"M": (M * link).astype(np.float32), "link": link}


def _link(L, rng, p_down, p_up):
    """Two-state telemetry chain: p_down to drop while up, p_up to recover while down."""
    link = np.ones(L)
    up = True
    u = rng.random(L)
    for k in range(L):
        up = (u[k] >= p_down) if up else (u[k] < p_up)
        link[k] = up
    return link


def _health_dead(r, rng, a=0.05, b=0.85, persist=0.7):
    """Sensor dropout whose hazard grows as remaining life falls, and which persists once dead."""
    q = a + b * (1.0 - np.asarray(r, dtype=np.float64)) ** 2
    u = rng.random(len(q))
    dead = np.zeros(len(q))
    for k in range(len(q)):
        p = max(persist, q[k]) if (k and dead[k - 1]) else q[k]
        dead[k] = u[k] < p
    return dead


def full(r, rng):
    return _finish(np.ones(len(r)), np.ones(len(r)))


def mixture(r, rng):
    L = len(r)
    link = _link(L, rng, 0.02, 0.10)
    return _finish((rng.random(L) < 0.6) * (1.0 - _health_dead(r, rng)), link)


def sparse_shift(r, rng):
    L = len(r)
    return _finish(rng.random(L) < 0.3, _link(L, rng, 0.06, 0.10))


PROCESSES = {"full": full, "mixture": mixture, "sparse_shift": sparse_shift}


def draw(name, ruls, names, n_rep, seed):
    """Reproducible replicates: list (per asset) of list (per replicate) of dict(M, link).

    The seed stream is keyed by the process, the asset name and the replicate, so a given
    (seed, asset, replicate) yields the same mask however the assets are ordered or subset.
    """
    proc = PROCESSES[name]
    key = sum(map(ord, name))
    out = []
    for r, asset_name in zip(ruls, names):
        rng = np.random.default_rng([seed, key, sum(map(ord, asset_name))])
        out.append([proc(r, rng) for _ in range(n_rep)])
    return out
