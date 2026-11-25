import math

import numpy as np
import pytest

from certalarm import baselines as B


def test_lbc_quantile_pools_time_points():
    q = B.lbc_quantile([np.array([0.5, 0.5])], [np.array([0.1, 0.1])], [np.array([0.4, 0.3])], 0.5)
    assert q == pytest.approx(2.0)


def test_sup_score_and_band():
    assert B.sup_score([0.5, 0.5], [0.1, 0.1], [0.4, 0.3]) == pytest.approx(2.0)
    assert B.sup_score([0.5, 0.5], [0.1, 0.1], [0.4, 0.3], can_fire=[1, 0]) == pytest.approx(1.0)
    assert B.supband_quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_lower_signal_with_infinite_quantile_fires_immediately():
    sig = B.lower_signal(np.array([0.9, 0.8]), np.array([0.1, 0.1]), math.inf)
    assert np.all(np.isneginf(sig))


def test_cost_optimal():
    late = np.array([[1.0, 0.0, 0.0]])
    rem = np.array([[0.0, 0.2, 0.9]])
    assert B.cost_optimal_threshold(late, rem, np.array([0.1, 0.2, 0.3]), c_late=1.0, c_waste=0.5) == 0.2


def test_zero_sigma_raises_instead_of_hiding_alarms():
    with pytest.raises(ValueError):
        B.sup_score([0.5, 0.5], [0.1, 0.0], [0.4, 0.3])
    with pytest.raises(ValueError):
        B.lbc_quantile([np.array([0.5])], [np.array([0.0])], [np.array([0.4])], 0.5)
