"""Tests for minute-grid features and fold-local scaling."""
import numpy as np
import pytest

from certalarm import features


def _sine(n_win, samples=256, amp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 25600.0
    return (amp * np.sin(2 * np.pi * 120.0 * t) + 0.01 * rng.standard_normal((n_win, samples))
            ).astype(np.float32)


def test_window_features_return_eighteen_finite_columns():
    f = features.window_features(_sine(5).astype(np.float64))
    assert f.shape == (5, 18)
    assert np.all(np.isfinite(f))


def test_log_rms_rises_with_amplitude():
    quiet = features.window_features(_sine(4, amp=0.1).astype(np.float64))
    loud = features.window_features(_sine(4, amp=10.0).astype(np.float64))
    assert loud[:, 0].mean() > quiet[:, 0].mean()


def test_minute_features_average_the_windows_of_each_minute():
    w = _sine(12)
    g = features.minute_features(w, per_min=6)
    assert g.shape == (2, 18)
    direct = features.window_features(w[:6].astype(np.float64)).mean(0)
    assert np.allclose(g[0], direct, atol=1e-5)


def test_minute_features_drop_a_partial_final_minute():
    assert features.minute_features(_sine(14), per_min=6).shape == (2, 18)


def test_baseline_correction_zeroes_the_commissioning_mean():
    g = np.tile(np.arange(18, dtype=np.float32), (30, 1))
    assert np.allclose(features.baseline_correct(g)[:features.BASELINE_MIN].mean(0), 0.0)


def test_scale_is_fitted_only_on_the_grids_it_is_given():
    small = np.random.default_rng(0).standard_normal((30, 18)).astype(np.float32)
    large = small * 100.0
    assert np.all(features.fit_scale([small]) < features.fit_scale([small, large]))


def test_normalize_is_clipped_and_float32():
    g = np.zeros((30, 18), dtype=np.float32)
    g[20] = 1e6
    out = features.normalize(g, np.ones(18, dtype=np.float32))
    assert out.dtype == np.float32
    assert out.max() <= 10.0 and out.min() >= -10.0


def test_asset_grid_is_cached_and_reused(tmp_path):
    w = _sine(12)
    a = features.asset_grid("pronostia", "BearingT_1", w, per_min=6, cache_dir=tmp_path)
    assert (tmp_path / "pronostia_BearingT_1.npy").exists()
    b = features.asset_grid("pronostia", "BearingT_1", None, per_min=6, cache_dir=tmp_path)
    assert np.array_equal(a, b)


def test_a_grid_shorter_than_the_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline"):
        features.baseline_correct(np.zeros((3, 18), dtype=np.float32))
