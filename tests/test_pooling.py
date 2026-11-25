"""Tests for condition-scaled CRC.

The guarantee rests on two things: the loss stays non-increasing in theta, and the scales come
only from training assets. Both are tested directly, because a violation of either is silent -
it produces a threshold that looks fine and is too loose.
"""
import zlib

import numpy as np
import pytest

from certalarm import certify, pooling


class _Traj:
    def __init__(self, mu, rul, sigma=None, can_fire=None):
        self.mu = np.asarray(mu, float)
        self.rul = np.asarray(rul, float)
        self.sigma = np.full(len(self.mu), 0.1) if sigma is None else np.asarray(sigma, float)
        self.can_fire = np.ones(len(self.mu), bool) if can_fire is None else can_fire


def _linear(n=100, bias=0.0):
    rul = np.linspace(1.0, 0.0, n)
    return _Traj(rul + bias, rul)


def _fit(spec, bias_of=lambda c: 0.0):
    return {name: {"condition": c, "reps": [_linear(bias=bias_of(c))]}
            for name, c in spec.items()}


def test_the_scale_is_mu_where_true_life_reaches_lead():
    t = _linear(n=101)                      # rul = 1.00, 0.99, ... ; mu == rul
    assert pooling.scale_at_lead(t, 0.05) == pytest.approx(0.05, abs=1e-9)


def test_a_truncated_record_falls_back_to_its_last_minute():
    t = _Traj(mu=[0.9, 0.8, 0.7], rul=[0.9, 0.8, 0.7])      # never reaches lead
    assert pooling.scale_at_lead(t, 0.05) == pytest.approx(0.7)


def test_scales_are_normalized_to_a_fleet_median_of_one():
    fit = _fit({"a": 1, "b": 2, "c": 3}, bias_of=lambda c: 0.1 * c)
    scales = pooling.condition_scales(fit, lead=0.05)
    assert np.median(list(scales.values())) == pytest.approx(1.0)


def test_a_condition_where_the_predictor_reads_high_gets_a_higher_scale():
    fit = _fit({"a": 1, "b": 2}, bias_of=lambda c: 0.0 if c == 1 else 0.4)
    scales = pooling.condition_scales(fit, lead=0.05)
    assert scales[2] > scales[1]


def test_scales_from_a_non_training_asset_are_refused():
    # The guarantee dies quietly if a calibration asset leaks into the scale estimate, so this
    # is a hard error rather than a warning.
    fit = _fit({"train_a": 1, "calib_b": 2})
    with pytest.raises(ValueError, match="training assets"):
        pooling.condition_scales(fit, lead=0.05, fit_names=["train_a"])


def test_the_declared_training_set_is_accepted():
    fit = _fit({"train_a": 1, "train_b": 2})
    scales = pooling.condition_scales(fit, lead=0.05, fit_names=["train_a", "train_b", "train_c"])
    assert set(scales) == {1, 2}


def test_an_unseen_condition_falls_back_to_scale_one():
    sig = pooling.signal_for({1: 2.0}, condition=99)
    t = _linear()
    assert np.allclose(sig(t), t.mu)


def test_a_nonpositive_scale_never_reaches_the_rule():
    # A negative divisor would flip the inequality and make the loss increase in theta, which
    # would violate the monotonicity CRC needs.
    sig = pooling.signal_for({1: -3.0}, condition=1)
    t = _linear()
    assert np.allclose(sig(t), t.mu)


def test_scaling_keeps_the_loss_non_increasing_in_theta():
    """The property certify._check enforces, on the scaled signal rather than the raw one."""
    from certalarm.decision import outcome_curves
    thetas = np.linspace(0.0, 1.0, 51)
    t = _linear()
    for scale in (0.5, 1.0, 2.0):
        sig = pooling.signal_for({1: scale}, condition=1)
        late, _ = outcome_curves(sig(t), t.rul, thetas, 0.05, t.can_fire)
        assert np.all(np.diff(late) <= 1e-12)
        certify._check(np.atleast_2d(late), thetas)      # raises if the property is broken


def test_the_floor_is_the_pooled_one_not_the_stratified_one():
    # The whole reason the scale lives in the rule: 17 calibration assets keep a floor of 0.056,
    # where three strata of about six would each be stuck at 0.143 and could not certify 0.10.
    assert certify.certifiable_floor(17) < 0.10 < certify.certifiable_floor(6)


def test_within_condition_spread_is_smaller_when_conditions_differ():
    fit = _fit({f"b{i}": c for i, c in enumerate([1, 1, 1, 2, 2, 2, 3, 3, 3])},
               bias_of=lambda c: 0.3 * c)
    pooled, within = pooling.spread_within_and_pooled(fit, lead=0.05)
    assert within < pooled


def test_spread_is_unchanged_when_the_condition_explains_nothing():
    # Same bias in every condition: grouping buys nothing and the two spreads should agree.
    fit = {f"b{i}": {"condition": i % 3, "reps": [_linear(bias=0.05 * i)]} for i in range(9)}
    pooled, within = pooling.spread_within_and_pooled(fit, lead=0.05)
    assert within == pytest.approx(pooled, rel=0.6)


def test_a_single_asset_condition_does_not_fake_a_perfect_fit():
    # One asset per condition would give zero residual and imply the grouping explains
    # everything; such conditions must contribute no degrees of freedom instead.
    fit = {f"b{i}": {"condition": i, "reps": [_linear(bias=0.1 * i)]} for i in range(5)}
    pooled, within = pooling.spread_within_and_pooled(fit, lead=0.05)
    assert within == pytest.approx(pooled)


# --- wiring: the scale must travel with the trajectory it belongs to -------------------------

class _Asset:
    def __init__(self, name, condition, n_min=40, per_min=6, seed=0):
        rng = np.random.default_rng(seed)
        self.name, self.condition, self.per_min = name, condition, per_min
        self.dataset = "pronostia"
        self.windows = rng.standard_normal((n_min * per_min, 8)).astype(np.float32)

    @property
    def n_min(self):
        return len(self.windows) // self.per_min

    @property
    def rul(self):
        return np.linspace(1.0, 0.0, self.n_min, dtype=np.float32)


class _Predictor:
    """Per-condition offset plus a fit-set-dependent bias.

    The bias matters: a fake whose output ignores the model would make every inner split compute
    identical scales, and a test built on it could not tell a per-model scale from a fold-level
    one. Real predictors depend on their fit set, so this one does too.
    """

    name = "fake"

    def fit(self, fit_assets, seed, process):
        names = sorted(a.name for a in fit_assets)
        return {"fit": names, "bias": (zlib.crc32("|".join(names).encode()) % 13) / 50.0}

    def predict(self, model, asset, M):
        mu = asset.rul.astype(float) + 0.1 * asset.condition + model["bias"]
        return mu, np.full(asset.n_min, 0.05)


def _fleet():
    return [_Asset(f"Bearing{c}_{i}", c, seed=10 * c + i)
            for c in (1, 2, 3) for i in range(1, 4)]


def _fold():
    from certalarm import crossfit
    fleet = _fleet()
    spec = crossfit.build_fold(fleet, seed=1)[0]
    return fleet, spec, crossfit.run_fold(fleet, spec, _Predictor(), "full", 1, 1)


def test_every_trajectory_carries_a_positive_scale():
    _, _, res = _fold()
    for reps in list(res.calibration.values()) + list(res.test.values()):
        assert all(t.scale > 0 for t in reps)


def test_a_test_asset_gets_one_scale_per_inner_model():
    # Each test trajectory is scored by one inner model and must use that model's scales.
    # If the scales were fold-level, every trajectory of an asset would share one value.
    _, _, res = _fold()
    scales = {name: {t.scale for t in reps} for name, reps in res.test.items()}
    assert any(len(v) > 1 for v in scales.values()), \
        "all test trajectories share a scale - the per-inner-model scale is not being applied"


def test_conditions_receive_different_scales():
    _, _, res = _fold()
    by_condition = {}
    for name, reps in res.calibration.items():
        by_condition.setdefault(int(name[7]), set()).update(t.scale for t in reps)
    assert len(by_condition) > 1 and len({min(v) for v in by_condition.values()}) > 1


def test_crc_scaled_is_fitted_and_reduces_to_crc_when_scales_are_one():
    from certalarm import evaluate
    _, _, res = _fold()
    rules = evaluate.fit_rules(res.calibration, 0.10, 0.05)
    assert "crc_scaled" in rules and rules["crc_scaled"]["scaled"] is True

    # Force every scale to 1: the scaled rule must then be the pooled rule exactly.
    for reps in res.calibration.values():
        for t in reps:
            t.scale = 1.0
    same = evaluate.fit_rules(res.calibration, 0.10, 0.05)
    assert same["crc_scaled"]["theta"] == same["crc"]["theta"]


def test_crc_scaled_keeps_the_pooled_floor():
    from certalarm import evaluate
    _, _, res = _fold()
    rules = evaluate.fit_rules(res.calibration, 0.10, 0.05)
    # Same calibration set, so the same floor - not the stratified 1/(n_k+1).
    assert rules["crc_scaled"]["floor"] == rules["crc"]["floor"]
    assert rules["crc_scaled"]["n_calibration"] == rules["crc"]["n_calibration"]


def test_a_scale_is_never_estimated_from_a_calibration_asset():
    # The guarantee dies silently if it is; condition_scales must refuse the whole call.
    from certalarm import crossfit
    fleet = _fleet()
    spec = crossfit.build_fold(fleet, seed=1)[0]
    held = set(spec["inner"][0]["held"])
    assert held, "fixture produced no held assets"
    assert not (held & set(spec["inner"][0]["fit"])), "fit and held overlap in the fixture"
