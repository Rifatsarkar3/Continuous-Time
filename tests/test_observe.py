"""Tests for observation processes on the minute grid."""
import numpy as np
import pytest

from certalarm import observe


def _rul(n=300):
    return np.linspace(1.0, 0.0, n, dtype=np.float32)


@pytest.mark.parametrize("name", sorted(observe.PROCESSES))
def test_every_process_returns_binary_arrays_of_the_right_length(name):
    r = _rul()
    out = observe.PROCESSES[name](r, np.random.default_rng(0))
    assert set(out) == {"M", "link"}
    for k in ("M", "link"):
        assert out[k].shape == r.shape
        assert out[k].dtype == np.float32
        assert set(np.unique(out[k])) <= {0.0, 1.0}


@pytest.mark.parametrize("name", sorted(observe.PROCESSES))
def test_commissioning_minutes_are_always_observed(name):
    out = observe.PROCESSES[name](_rul(), np.random.default_rng(3))
    assert np.all(out["M"][:observe.BASELINE_MIN] == 1.0)
    assert np.all(out["link"][:observe.BASELINE_MIN] == 1.0)


@pytest.mark.parametrize("name", sorted(observe.PROCESSES))
def test_nothing_is_acquired_while_the_link_is_down(name):
    out = observe.PROCESSES[name](_rul(), np.random.default_rng(5))
    assert not np.any((out["M"] > 0) & (out["link"] == 0))


def test_full_observes_every_minute():
    out = observe.PROCESSES["full"](_rul(), np.random.default_rng(0))
    assert np.all(out["M"] == 1.0)


def test_sparse_shift_observes_far_less_than_full():
    out = observe.PROCESSES["sparse_shift"](_rul(2000), np.random.default_rng(1))
    assert 0.1 < out["M"].mean() < 0.4


def test_mixture_dropout_worsens_near_failure():
    r = _rul(4000)
    rates = [observe.PROCESSES["mixture"](r, np.random.default_rng(s))["M"] for s in range(8)]
    m = np.mean(rates, axis=0)
    assert m[-500:].mean() < m[observe.BASELINE_MIN:1000].mean()


def test_draw_is_reproducible_and_differs_between_assets():
    r = [_rul(200), _rul(200)]
    a = observe.draw("mixture", r, ["B1", "B2"], n_rep=3, seed=7)
    b = observe.draw("mixture", r, ["B1", "B2"], n_rep=3, seed=7)
    assert len(a) == 2 and len(a[0]) == 3
    assert np.array_equal(a[0][0]["M"], b[0][0]["M"])
    assert not np.array_equal(a[0][0]["M"], a[1][0]["M"])


def test_draw_replicates_within_an_asset_differ():
    a = observe.draw("sparse_shift", [_rul(200)], ["B1"], n_rep=2, seed=7)
    assert not np.array_equal(a[0][0]["M"], a[0][1]["M"])


def test_unknown_process_is_rejected():
    with pytest.raises(KeyError):
        observe.draw("nonexistent", [_rul(50)], ["B1"], n_rep=1, seed=0)
