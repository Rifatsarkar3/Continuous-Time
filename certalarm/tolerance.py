"""One-sided parametric tolerance limits: the normal, lognormal and Weibull counterparts of the distribution-free limit.

A lower limit has at least `content` of the population above it, an upper limit at least `content` below it, each with
probability `confidence` over samples if the family is right. Samples are rows; all functions work on many at once.
"""
from functools import lru_cache

import numpy as np
from scipy import stats as st


def normal_factor(n, content, confidence):
    """Exact factor: mean +/- factor * sd is the one-sided normal tolerance limit."""
    return float(st.nct.ppf(confidence, n - 1, st.norm.ppf(content) * np.sqrt(n)) / np.sqrt(n))


def normal_limit(x, content, confidence, upper):
    x = np.atleast_2d(np.asarray(x, float))
    k = normal_factor(x.shape[1], content, confidence)
    return x.mean(axis=1) + (k if upper else -k) * x.std(axis=1, ddof=1)


def lognormal_limit(x, content, confidence, upper):
    return np.exp(normal_limit(np.log(x), content, confidence, upper))


def sev_fit(y):
    """Maximum likelihood location and scale of the smallest extreme value law (log-Weibull), row by row."""
    y = np.atleast_2d(np.asarray(y, float))
    m, s = y.mean(axis=1, keepdims=True), y.std(axis=1, keepdims=True)
    z = (y - m) / s
    k = np.full((len(z), 1), 1.28)                      # 1 / scale of standardised data
    for _ in range(100):
        e = np.exp(k * z)
        a, b, c = e.sum(1, keepdims=True), (z * e).sum(1, keepdims=True), (z * z * e).sum(1, keepdims=True)
        step = (b / a - 1 / k) / (c / a - (b / a) ** 2 + 1 / k ** 2)
        k = np.maximum(k - step, k / 4)
        if np.max(np.abs(step) / k) < 1e-12:
            break
    scale = 1 / k
    loc = scale * np.log(np.exp(k * z).mean(axis=1, keepdims=True))
    return (m + s * loc).ravel(), (s * scale).ravel()


@lru_cache(maxsize=None)
def _weibull_pivot(n, p, draws, seed):
    y = np.log(np.random.default_rng(seed).exponential(size=(draws, n)))
    u, b = sev_fit(y)
    w = np.log(-np.log1p(-p))
    return np.sort((u + b * w - w) / b)                 # (estimated quantile - true quantile) / estimated scale


def weibull_limit(x, content, confidence, upper, draws=20_000, seed=0):
    x = np.atleast_2d(np.asarray(x, float))
    p = content if upper else 1 - content
    u, b = sev_fit(np.log(x))
    q = np.quantile(_weibull_pivot(x.shape[1], round(p, 12), draws, seed), 1 - confidence if upper else confidence)
    return np.exp(u + b * (np.log(-np.log1p(-p)) - q))
