"""Tests for the LeadPoint alarm: envelope features, exact certification, and the scorers."""
import numpy as np
import pytest

from certalarm import certify, leadpoint

FS = 25600.0


def _am(n_win, carrier=3000.0, mod=160.0, depth=0.8, samples=2560, seed=0):
    """Amplitude-modulated carrier: its envelope spectrum peaks at the modulation frequency."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / FS
    x = (1 + depth * np.cos(2 * np.pi * mod * t)) * np.sin(2 * np.pi * carrier * t)
    return (x + 0.01 * rng.standard_normal((n_win, samples))).astype(np.float32)


def test_envelope_band_features_return_nine_finite_columns():
    f = leadpoint.envelope_band_features(_am(4))
    assert f.shape == (4, len(leadpoint.ENV_NAMES)) == (4, 9)
    assert np.all(np.isfinite(f))


def test_modulation_frequency_lands_in_its_envelope_band():
    f = leadpoint.envelope_band_features(_am(3, mod=160.0))
    band = int(np.searchsorted(leadpoint.ENV_EDGES, 160.0)) - 1
    assert np.all(np.argmax(f[:, :8], axis=1) == band)


def test_minute_env_features_average_the_windows_of_each_minute():
    w = _am(14)
    g = leadpoint.minute_env_features(w, per_min=6)
    assert g.shape == (2, 9)
    assert np.allclose(g[0], leadpoint.envelope_band_features(w[:6]).mean(0), atol=1e-5)


def test_ewma_is_causal_and_leaves_a_constant_unchanged():
    x = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float32)
    y = leadpoint.ewma(x, halflife=5.0)
    x2 = x.copy()
    x2[30:] += 100.0
    assert np.allclose(leadpoint.ewma(x2, 5.0)[:30], y[:30])
    c = np.full((20, 2), 3.0, np.float32)
    assert np.allclose(leadpoint.ewma(c, 5.0), 3.0)


def test_lead_point_is_the_running_minimum_at_the_last_acceptable_step():
    s = np.array([0.9, 0.4, 0.7, 0.2, 0.1])
    rul = np.linspace(1, 0, 5)                 # 1, .75, .5, .25, 0
    assert leadpoint.lead_point(s, rul, 0.3) == pytest.approx(0.4)
    assert leadpoint.lead_point(s, rul, 2.0) == -np.inf


@pytest.mark.parametrize("n", [9, 12, 13, 20, 40])
def test_exact_threshold_is_the_smallest_theta_meeting_the_conformal_bound(n):
    alpha = 0.10
    s = np.random.default_rng(n).uniform(0, 1, n)
    th = leadpoint.exact_threshold(s, alpha)
    risk = lambda t: (np.sum(s >= t) + 1) / (n + 1)
    assert risk(th) <= alpha + 1e-12
    below = np.max(s[s < th])                  # the next score down fails the bound
    assert risk(below) > alpha


def test_exact_threshold_never_exceeds_the_grid_crc_threshold():
    s = np.random.default_rng(3).uniform(0, 1, 13)
    thetas = np.linspace(0, 1, 201)
    L = np.stack([(thetas <= v).astype(float) for v in s])
    grid = certify.crc_threshold(L, thetas, 0.10)
    exact = leadpoint.exact_threshold(s, 0.10)
    assert exact <= grid + 1e-12
    assert grid - exact <= thetas[1] - thetas[0] + 1e-12


def test_exact_threshold_declines_below_the_certifiability_floor():
    assert leadpoint.exact_threshold(np.random.default_rng(0).uniform(0, 1, 8), 0.10) is None


def _synthetic_fleet(seed, n_assets=6):
    """Feature 0 drops in the last 15% of life; feature 1 is pure noise."""
    rng = np.random.default_rng(seed)
    Xs, ruls = [], []
    for _ in range(n_assets):
        n = int(rng.integers(40, 120))
        r = np.linspace(1, 0, n).astype(np.float32)
        x0 = np.where(r < 0.15, -3.0 * (0.15 - r) / 0.15, 0.0) + 0.1 * rng.standard_normal(n)
        x1 = rng.standard_normal(n)
        Xs.append(np.stack([x0, x1], 1).astype(np.float32))
        ruls.append(r)
    return Xs, ruls


@pytest.mark.parametrize("hidden", [0, 16])
def test_rank_scorer_puts_lead_point_states_below_early_states(hidden):
    Xs, ruls = _synthetic_fleet(0)
    model = leadpoint.fit_rank(Xs, ruls, lead=0.05, hidden=hidden, iters=200, seed=0)
    Xt, rt = _synthetic_fleet(1)
    for x, r in zip(Xt, rt):
        s = leadpoint.score(model, x)
        assert leadpoint.lead_point(s, r, 0.05) < np.min(s[r >= 0.5])


def test_ridge_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    X = [rng.standard_normal((200, 3)).astype(np.float32)]
    y = [(0.5 + 0.2 * X[0][:, 0] - 0.1 * X[0][:, 2]).astype(np.float32)]
    m = leadpoint.fit_ridge(X, y, lam=1e-6)
    assert np.allclose(leadpoint.score(m, X[0]), y[0], atol=1e-3)


def test_ridge_module_reproduces_the_ridge_scores_and_is_hashable_by_the_registry():
    from certalarm import registry
    rng = np.random.default_rng(1)
    X = [rng.standard_normal((80, 4)).astype(np.float32)]
    y = [rng.uniform(0, 1, 80).astype(np.float32)]
    m = leadpoint.fit_ridge(X, y, lam=1.0)
    scale = np.full(4, 2.0, np.float32)
    mod = leadpoint.ridge_module(m, scale, resid_sd=0.1)
    assert np.allclose(leadpoint.module_score(mod, X[0]), leadpoint.score(m, X[0]), atol=1e-12)
    assert np.allclose(mod.scale.numpy(), scale) and float(mod.resid_sd) == 0.1
    h = registry.weights_hash(mod)
    m2 = dict(m, w=m["w"] + 1e-9)
    assert registry.weights_hash(leadpoint.ridge_module(m2, scale, 0.1)) != h
