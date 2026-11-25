import numpy as np

from certalarm import decision as D


def test_first_crossing_basic():
    s = np.array([0.9, 0.5, 0.2, 0.1])
    assert D.first_crossing(s, 0.3) == 2
    assert D.first_crossing(s, 0.05) is None
    assert D.first_crossing(s, 0.3, can_fire=np.array([1, 1, 0, 1])) == 3


def test_alarm_outcome_late_and_on_time():
    s = np.array([0.9, 0.5, 0.2, 0.1])
    r = np.array([0.9, 0.6, 0.3, 0.0])
    assert D.alarm_outcome(s, r, 0.3, lead=0.05) == (False, 0.3)
    assert D.alarm_outcome(s, r, 0.15, lead=0.05) == (True, 0.0)
    assert D.alarm_outcome(s, r, 0.01, lead=0.05) == (True, 0.0)


def test_curves_match_scalar_and_are_monotone():
    rng = np.random.default_rng(1)
    thetas = np.round(np.arange(0, 1.01, 0.05), 2)
    for _ in range(200):
        L = int(rng.integers(5, 60))
        s = rng.random(L)
        r = np.sort(rng.random(L))[::-1]
        c = rng.random(L) < 0.6
        late, rem = D.outcome_curves(s, r, thetas, 0.1, c)
        assert np.all(np.diff(late) <= 0)
        for j, th in enumerate(thetas):
            lt, rm = D.alarm_outcome(s, r, th, 0.1, c)
            assert late[j] == float(lt) and rem[j] == rm


def test_asset_curves_average_replicates():
    thetas = np.array([0.1, 0.5])
    r = np.array([1.0, 0.5, 0.0])
    reps = [(np.array([0.9, 0.4, 0.0]), r, None), (np.array([0.9, 0.9, 0.0]), r, None)]
    late, rem = D.asset_curves(reps, thetas, lead=0.05)
    assert np.allclose(late, [1.0, 0.5]) and np.allclose(rem, [0.0, 0.25])


def test_outcome_curves_empty_replicate_is_always_late():
    thetas = np.array([0.1, 0.5, 0.9])
    late, rem = D.outcome_curves(np.array([]), np.array([]), thetas, lead=0.1)
    assert np.all(late == 1.0) and np.all(rem == 0.0)
    for th in thetas:
        assert D.alarm_outcome(np.array([]), np.array([]), th, lead=0.1) == (True, 0.0)


def test_asset_curves_tolerates_empty_replicate():
    thetas = np.array([0.1, 0.5])
    r = np.array([1.0, 0.0])
    reps = [(np.array([0.9, 0.0]), r, None), (np.array([]), np.array([]), None)]
    late, rem = D.asset_curves(reps, thetas, lead=0.05)
    assert np.allclose(late, [1.0, 1.0])
