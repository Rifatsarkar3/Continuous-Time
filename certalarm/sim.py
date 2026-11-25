"""Synthetic heterogeneous run-to-failure fleet (library validation, E0).

Unit life T ~ Weibull(shape, scale set so the mean is mean_life); normalized RUL
r_t = 1 - t/(T-1). The predictor output is mu_t = clip(r_t + b + e_t, 0, 1), where b ~
N(0, bias_sd) is a unit-level bias (asset heterogeneity) and e_t is stationary AR(1) noise.
The predictive sigma is the constant noise_sd; it deliberately ignores the unit bias, as
real per-time uncertainty heads do.
"""
import math

import numpy as np
from scipy.signal import lfilter


def simulate_unit(rng, mean_life=200.0, shape=2.0, bias_sd=0.08, noise_sd=0.04, ar=0.9, min_life=20):
    scale = mean_life / math.gamma(1 + 1 / shape)
    T = max(min_life, int(round(scale * rng.weibull(shape))))
    r = 1.0 - np.arange(T) / (T - 1)
    w = rng.normal(0.0, noise_sd * math.sqrt(1 - ar ** 2), T)
    w[0] = rng.normal(0.0, noise_sd)
    e = lfilter([1.0], [1.0, -ar], w)
    mu = np.clip(r + rng.normal(0.0, bias_sd) + e, 0.0, 1.0)
    return {"rul": r, "mu": mu, "sigma": np.full(T, noise_sd)}


def simulate_fleet(n_units, rng, **kw):
    return [simulate_unit(rng, **kw) for _ in range(n_units)]
