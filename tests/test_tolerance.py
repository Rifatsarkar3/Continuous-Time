"""Parametric tolerance limits: table values, and the confidence they claim on their own family."""
import numpy as np
import pytest
from scipy import stats as st

from certalarm import tolerance as TL


@pytest.mark.parametrize("n, content, confidence, k", [(10, 0.90, 0.95, 2.355), (10, 0.95, 0.95, 2.911), (20, 0.90, 0.95, 1.926)])
def test_normal_factor_matches_published_tables(n, content, confidence, k):
    assert TL.normal_factor(n, content, confidence) == pytest.approx(k, abs=6e-3)


def test_sev_fit_matches_scipy():
    x = np.random.default_rng(3).weibull(2.5, 40) * 7
    c, _, scale = st.weibull_min.fit(x, floc=0)
    u, b = TL.sev_fit(np.log(x))
    assert b[0] == pytest.approx(1 / c, rel=1e-4) and np.exp(u[0]) == pytest.approx(scale, rel=1e-4)


@pytest.mark.parametrize("n", [9, 36])
@pytest.mark.parametrize("upper", [False, True])
def test_limits_hold_their_confidence_on_their_own_family(n, upper):
    rng = np.random.default_rng(11)
    x = 5 * rng.weibull(1.7, (8000, n))
    lim = TL.weibull_limit(x, 0.90, 0.975, upper)
    content = np.exp(-(lim / 5) ** 1.7)                    # P(X > limit)
    content = 1 - content if upper else content
    assert np.mean(content >= 0.90) == pytest.approx(0.975, abs=0.007)
    z = rng.normal(2, 3, (8000, n))
    lim = TL.normal_limit(z, 0.90, 0.975, upper)
    content = st.norm.sf((lim - 2) / 3)
    content = 1 - content if upper else content
    assert np.mean(content >= 0.90) == pytest.approx(0.975, abs=0.007)
    assert np.allclose(TL.lognormal_limit(np.exp(z), 0.90, 0.975, upper), np.exp(lim))


def test_weibull_limit_is_scale_equivariant():
    x = np.random.default_rng(5).weibull(2.0, (3, 9))
    assert np.allclose(TL.weibull_limit(10 * x, 0.9, 0.975, False), 10 * TL.weibull_limit(x, 0.9, 0.975, False))
