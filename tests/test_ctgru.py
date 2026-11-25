"""Tests for the CT-GRU predictor module."""
import numpy as np
import pytest
import torch

from certalarm.predictors.ctgru import CTGRU


def _batch(B=2, L=7, F=18, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, L, F, generator=g)
    M = torch.ones(B, L)
    return X, M


def test_output_shapes_and_ranges():
    X, M = _batch()
    mu, sigma = CTGRU(18)(X, M)
    assert mu.shape == (2, 7) and sigma.shape == (2, 7)
    assert torch.all((mu >= 0) & (mu <= 1))
    assert torch.all(sigma > 0)


def test_an_unobserved_minute_ignores_its_input():
    torch.manual_seed(0)
    m = CTGRU(18).eval()
    X, M = _batch()
    M[:, 3] = 0.0
    X2 = X.clone()
    X2[:, 3] = 999.0
    with torch.no_grad():
        a, _ = m(X, M)
        b, _ = m(X2, M)
    assert torch.allclose(a, b, atol=1e-6)


def test_an_observed_minute_does_use_its_input():
    torch.manual_seed(0)
    m = CTGRU(18).eval()
    X, M = _batch()
    X2 = X.clone()
    X2[:, 3] = 5.0
    with torch.no_grad():
        a, _ = m(X, M)
        b, _ = m(X2, M)
    assert not torch.allclose(a, b, atol=1e-6)


def test_the_state_still_drifts_across_an_unobserved_gap():
    torch.manual_seed(0)
    m = CTGRU(18).eval()
    X, M = _batch(B=1, L=6)
    M[:, 1:] = 0.0
    with torch.no_grad():
        mu, _ = m(X, M)
    assert not torch.allclose(mu[0, 1:], mu[0, 1].expand(5), atol=1e-6)


def test_predictions_are_causal():
    torch.manual_seed(0)
    m = CTGRU(18).eval()
    X, M = _batch(B=1, L=8)
    X2 = X.clone()
    X2[:, 6:] = 50.0
    with torch.no_grad():
        a, _ = m(X, M)
        b, _ = m(X2, M)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)


def test_gradients_reach_every_parameter():
    X, M = _batch()
    m = CTGRU(18)
    mu, sigma = m(X, M)
    (mu.mean() + sigma.mean()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == []


def test_feature_width_is_respected():
    with pytest.raises(RuntimeError):
        CTGRU(18)(*_batch(F=7))


from certalarm.predictors import ctgru_train


def _toy_fleet(n=4, L=40, F=18, seed=0):
    rng = np.random.default_rng(seed)
    X = [rng.standard_normal((L, F)).astype(np.float32) for _ in range(n)]
    rul = [np.linspace(1, 0, L, dtype=np.float32) for _ in range(n)]
    return X, rul


def test_training_is_reproducible_for_a_seed():
    X, rul = _toy_fleet()
    a = ctgru_train.train(X, rul, "full", iters=5, seed=42)
    b = ctgru_train.train(X, rul, "full", iters=5, seed=42)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb)


def test_training_changes_the_weights():
    X, rul = _toy_fleet()
    before = CTGRU(18)
    after = ctgru_train.train(X, rul, "full", iters=5, seed=42)
    assert not all(torch.allclose(a, b)
                   for a, b in zip(before.parameters(), after.parameters()))


def test_predict_returns_one_value_per_minute():
    X, rul = _toy_fleet()
    m = ctgru_train.train(X, rul, "full", iters=2, seed=0)
    mu, sigma = ctgru_train.predict(m, X[0], np.ones(len(X[0]), dtype=np.float32))
    assert mu.shape == (40,) and sigma.shape == (40,)
    assert np.all(np.isfinite(mu)) and np.all(sigma > 0)


def test_predict_accepts_assets_of_different_lengths():
    X, rul = _toy_fleet()
    m = ctgru_train.train(X, rul, "full", iters=2, seed=0)
    short = X[0][:17]
    mu, _ = ctgru_train.predict(m, short, np.ones(17, dtype=np.float32))
    assert mu.shape == (17,)


def test_training_tolerates_assets_of_different_lengths():
    X, rul = _toy_fleet()
    X[1], rul[1] = X[1][:23], rul[1][:23]
    m = ctgru_train.train(X, rul, "full", iters=3, seed=1)
    assert isinstance(m, CTGRU)


def test_budget_is_fixed_per_dataset():
    assert ctgru_train.ITERS["pronostia"] == 150
    assert ctgru_train.ITERS["xjtu"] == 750


def test_the_default_loss_is_unchanged():
    # Every cached model was trained with the original loss; the defaults must reproduce it.
    import torch
    from certalarm.predictors import ctgru_train
    torch.manual_seed(0)
    mu, y = torch.rand(2, 7), torch.rand(2, 7)
    sigma, V = torch.rand(2, 7) + 0.1, torch.ones(2, 7)
    base = (mu - y) ** 2 + 0.1 * (0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma))
    want = ((base * V).sum(1) / V.sum(1)).mean()
    assert torch.allclose(ctgru_train._loss(mu, sigma, y, V), want)


def test_optimism_near_failure_costs_more_than_pessimism():
    # The failure mode the diagnosis found: reading high near failure makes the alarm late.
    import torch
    from certalarm.predictors import ctgru_train
    y = torch.full((1, 1), 0.05)
    sigma, V = torch.ones(1, 1), torch.ones(1, 1)
    high = ctgru_train._loss(y + 0.2, sigma, y, V, late_weight=4.0, asym=4.0)
    low = ctgru_train._loss(y - 0.2, sigma, y, V, late_weight=4.0, asym=4.0)
    assert high > 2 * low


def test_errors_near_failure_outweigh_the_same_error_in_healthy_life():
    import torch
    from certalarm.predictors import ctgru_train
    sigma, V = torch.ones(1, 1), torch.ones(1, 1)
    near = ctgru_train._loss(torch.full((1, 1), 0.15), sigma, torch.full((1, 1), 0.05), V,
                             late_weight=4.0)
    far = ctgru_train._loss(torch.full((1, 1), 0.90), sigma, torch.full((1, 1), 0.80), V,
                            late_weight=4.0)
    assert near > far


def test_a_non_default_loss_gets_its_own_cache_key():
    # Otherwise a new-loss model would be served a cached MSE model and the experiment would
    # silently compare the baseline against itself.
    from certalarm.predictors import modelcache
    base = modelcache.key("ctgru", "pronostia", 42, ["B1"], {"process": "full", "iters": 150})
    alarm = modelcache.key("ctgru", "pronostia", 42, ["B1"],
                           {"process": "full", "iters": 150,
                            "loss": {"late_weight": 4.0, "asym": 4.0}})
    assert base != alarm


def test_gating_leaves_early_life_asymmetry_out():
    # The flaw the first search exposed: ungated asymmetry penalises optimism in healthy life
    # too, which drives the whole trajectory down and fires the alarm early.
    import torch
    from certalarm.predictors import ctgru_train
    sigma, V = torch.ones(1, 1), torch.ones(1, 1)
    y_early = torch.full((1, 1), 0.80)
    over = y_early + 0.1
    ungated = ctgru_train._loss(over, sigma, y_early, V, asym=9.0, gate_asym=False)
    gated = ctgru_train._loss(over, sigma, y_early, V, asym=9.0, gate_asym=True)
    plain = ctgru_train._loss(over, sigma, y_early, V)
    assert ungated > gated
    assert abs(float(gated - plain)) < 1e-4      # early life is left essentially untouched


def test_gating_still_penalises_optimism_near_failure():
    import torch
    from certalarm.predictors import ctgru_train
    sigma, V = torch.ones(1, 1), torch.ones(1, 1)
    y_late = torch.full((1, 1), 0.02)
    over, under = y_late + 0.1, y_late - 0.1
    hi = ctgru_train._loss(over, sigma, y_late, V, asym=9.0, gate_asym=True)
    lo = ctgru_train._loss(under, sigma, y_late, V, asym=9.0, gate_asym=True)
    assert hi > 3 * lo


def test_gating_does_not_change_the_default_loss():
    import torch
    from certalarm.predictors import ctgru_train
    torch.manual_seed(0)
    mu, y = torch.rand(2, 5), torch.rand(2, 5)
    sigma, V = torch.rand(2, 5) + 0.1, torch.ones(2, 5)
    assert torch.allclose(ctgru_train._loss(mu, sigma, y, V, gate_asym=True),
                          ctgru_train._loss(mu, sigma, y, V, gate_asym=False))
