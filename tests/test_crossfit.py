"""Tests for nested cross-fitting."""
import numpy as np
import pytest

from certalarm import crossfit


class FakeAsset:
    def __init__(self, name, condition, n_min=40, per_min=6, samples=64, seed=0):
        rng = np.random.default_rng(seed)
        self.name, self.condition, self.per_min = name, condition, per_min
        self.dataset = "pronostia"
        self.windows = rng.standard_normal((n_min * per_min, samples)).astype(np.float32)

    @property
    def n_min(self):
        return len(self.windows) // self.per_min

    @property
    def rul(self):
        return np.linspace(1.0, 0.0, self.n_min, dtype=np.float32)


def _fleet(n_per_cond=4):
    return [FakeAsset(f"Bearing{c}_{i}", c, seed=10 * c + i)
            for c in (1, 2, 3) for i in range(1, n_per_cond + 1)]


class EchoPredictor:
    """Records which assets each model was trained on, and predicts the truth plus a bias."""

    name = "echo"

    def fit(self, fit_assets, seed, **kw):
        return {"trained_on": sorted(a.name for a in fit_assets), "seed": seed}

    def predict(self, model, asset, M):
        return asset.rul.astype(float) + 0.01, np.full(asset.n_min, 0.05)


def test_every_calibration_asset_gets_a_trajectory():
    a = _fleet()
    fold = crossfit.run_fold(a, crossfit.build_fold(a, seed=0)[0], EchoPredictor(),
                             process="full", n_rep=1, seed=0)
    assert set(fold.calibration) == set(fold.fold["calibration"])


def test_calibration_predictions_never_come_from_a_model_that_saw_the_asset():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, EchoPredictor(), process="full", n_rep=1, seed=0)
    for name, info in fold.calibration_sources.items():
        assert name not in info["trained_on"]


def test_no_model_in_the_fold_ever_saw_a_test_asset():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, EchoPredictor(), process="full", n_rep=1, seed=0)
    for m in fold.models:
        assert not set(spec["test"]) & set(m["trained_on"])


class BiasedPredictor:
    """Each inner model carries its own offset, so an averaged prediction would be
    distinguishable from any single model's."""

    name = "biased"

    def fit(self, fit_assets, seed, **kw):
        return {"trained_on": sorted(a.name for a in fit_assets), "bias": 0.1 * (seed % 10)}

    def predict(self, model, asset, M):
        return asset.rul.astype(float) + model["bias"], np.full(asset.n_min, 0.05)


def test_test_assets_get_a_trajectory_per_replicate_and_inner_model():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, EchoPredictor(), process="mixture", n_rep=3, seed=0)
    for name in spec["test"]:
        assert len(fold.test[name]) == 3 * len(fold.models)


def test_a_test_trajectory_comes_from_one_model_and_is_never_an_average():
    # Averaging predictions across inner models would hand the test asset a smoother signal
    # than any calibration asset ever saw, which breaks exchangeability in the direction that
    # makes the certified threshold too loose.
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, BiasedPredictor(), process="full", n_rep=1, seed=0)
    offsets = sorted({round(float(t.mu[0] - t.rul[0]), 6) for t in fold.test[spec["test"][0]]})
    assert offsets == sorted({round(m["bias"], 6) for m in fold.models})


def test_a_calibration_trajectory_and_a_test_trajectory_carry_the_same_kind_of_signal():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, BiasedPredictor(), process="full", n_rep=1, seed=0)
    cal_offsets = {round(float(t.mu[0] - t.rul[0]), 6)
                   for reps in fold.calibration.values() for t in reps}
    test_offsets = {round(float(t.mu[0] - t.rul[0]), 6)
                    for reps in fold.test.values() for t in reps}
    assert cal_offsets <= test_offsets


def test_trajectory_fields_line_up_in_length():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, EchoPredictor(), process="mixture", n_rep=2, seed=0)
    for reps in fold.test.values():
        for t in reps:
            n = len(t.rul)
            assert len(t.mu) == n and len(t.sigma) == n and len(t.can_fire) == n


def test_can_fire_follows_the_observation_process():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    full = crossfit.run_fold(a, spec, EchoPredictor(), process="full", n_rep=1, seed=0)
    sparse = crossfit.run_fold(a, spec, EchoPredictor(), process="sparse_shift", n_rep=1, seed=0)
    name = spec["test"][0]
    assert full.test[name][0].can_fire.all()
    assert not sparse.test[name][0].can_fire.all()


def test_the_commissioning_period_is_always_firable():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, EchoPredictor(), process="sparse_shift", n_rep=1, seed=0)
    for reps in fold.test.values():
        assert reps[0].can_fire[:10].all()


def test_the_run_is_reproducible_for_a_seed():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    x = crossfit.run_fold(a, spec, EchoPredictor(), process="mixture", n_rep=2, seed=7)
    y = crossfit.run_fold(a, spec, EchoPredictor(), process="mixture", n_rep=2, seed=7)
    n = spec["test"][0]
    assert np.array_equal(x.test[n][0].can_fire, y.test[n][0].can_fire)
    assert np.allclose(x.test[n][0].mu, y.test[n][0].mu)


def test_calibration_and_test_use_the_same_observation_process():
    a = _fleet()
    spec = crossfit.build_fold(a, seed=0)[0]
    fold = crossfit.run_fold(a, spec, EchoPredictor(), process="sparse_shift", n_rep=1, seed=0)
    cal_rate = np.mean([t.can_fire.mean() for reps in fold.calibration.values() for t in reps])
    test_rate = np.mean([t.can_fire.mean() for reps in fold.test.values() for t in reps])
    assert abs(cal_rate - test_rate) < 0.15
