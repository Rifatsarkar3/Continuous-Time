import math

import numpy as np
import pytest

from certalarm import certify as C


def test_floor_and_min_assets():
    assert C.certifiable_floor(9) == pytest.approx(0.1)
    assert C.min_assets(0.10) == 9
    assert C.min_assets(0.05) == 19


def test_crc_infeasible_below_floor():
    L = np.zeros((4, 3))
    th = np.array([0.1, 0.2, 0.3])
    assert C.crc_threshold(L, th, alpha=0.1) is None
    assert C.crc_threshold(L, th, alpha=0.25) == 0.1


def test_crc_picks_smallest_feasible_theta():
    thetas = np.array([0.1, 0.2, 0.3, 0.4])
    L = np.array([[1, 1, 0, 0]] * 2 + [[1, 0, 0, 0]] * 8, float)
    assert C.crc_threshold(L, thetas, alpha=0.1) == 0.3
    assert C.empirical_threshold(L, thetas, alpha=0.2) == 0.2


def test_rejects_non_monotone_losses():
    with pytest.raises(ValueError):
        C.crc_threshold(np.array([[0.0, 1.0]]), np.array([0.1, 0.2]), 0.5)


def test_conformal_quantile():
    s = np.arange(1, 10)
    assert C.conformal_quantile(s, 0.1) == 9.0
    assert C.conformal_quantile(s, 0.05) == math.inf


def test_conformal_quantile_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        C.conformal_quantile(np.array([1.0, np.nan, 3.0]), 0.1)


def test_rejects_unsorted_thetas():
    L = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        C.crc_threshold(L, np.array([0.3, 0.1, 0.2]), alpha=0.5)
    with pytest.raises(ValueError):
        C.crc_threshold(L, np.array([0.1, 0.2]), alpha=0.5)


def test_verification_floor_matches_the_exact_binomial_bound_with_no_events():
    from certalarm import stats
    for n in (5, 17, 30, 85):
        assert C.verification_floor(n) == pytest.approx(stats.clopper_pearson(0, n)[1])


def test_verification_floor_falls_as_the_fleet_grows():
    f = [C.verification_floor(n) for n in (5, 17, 30, 85, 709)]
    assert all(a > b for a, b in zip(f, f[1:]))


def test_the_two_floors_are_different_limits():
    # With 17 bearings a 10% risk is certifiable but cannot be shown to have been achieved.
    assert C.certifiable_floor(17) < 0.10 < C.verification_floor(17)


def test_min_assets_to_verify_is_the_smallest_fleet_that_could_show_the_risk():
    n = C.min_assets_to_verify(0.10)
    assert C.verification_floor(n) <= 0.10
    assert C.verification_floor(n - 1) > 0.10
