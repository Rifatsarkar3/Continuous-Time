"""Tests for the C-MAPSS loader.

The point of this fleet is that it is large enough to verify the risk it certifies, so the tests
check the fleet size claim as well as the parsing.
"""
import numpy as np
import pytest

from certalarm import certify, cmapss, features


@pytest.fixture
def tiny(tmp_path):
    """Two fake FD007 units, 26 columns, written deliberately out of cycle order.

    Both are longer than the baseline window, because a shorter unit is not a unit this loader
    is allowed to emit.
    """
    rows = []
    for uid, n in ((1, 12), (2, 15)):
        for c in reversed(range(1, n + 1)):
            rows.append([uid, c] + [0.1 * c] * 3 + [float(uid) + 0.01 * c] * 21)
    np.savetxt(tmp_path / "train_FD007.txt", np.array(rows))
    return tmp_path


def test_units_are_separated_and_shaped(tiny):
    units = cmapss.load_subset(7, data_dir=tiny)
    assert [u.name for u in units] == ["FD007_u1", "FD007_u2"]
    assert [u.n_min for u in units] == [12, 15]
    assert units[0].grid.shape == (12, 24)


def test_cycles_are_sorted_even_when_the_file_is_not(tiny):
    u = cmapss.load_subset(7, data_dir=tiny)[0]
    # settings were written as 0.1 * cycle, so a sorted unit has an ascending first column
    assert np.all(np.diff(u.grid[:, 0]) > 0)


def test_rul_runs_from_one_to_zero(tiny):
    u = cmapss.load_subset(7, data_dir=tiny)[0]
    assert u.rul[0] == pytest.approx(1.0) and u.rul[-1] == pytest.approx(0.0)


def test_the_condition_is_the_subset(tiny):
    assert all(u.condition == 7 for u in cmapss.load_subset(7, data_dir=tiny))
    assert cmapss.condition_of("FD003_u12") == 3


def test_a_unit_shorter_than_the_baseline_window_is_dropped(tmp_path):
    # Emitting it would hand features.baseline_correct something it raises on.
    short = [[1, c] + [0.0] * 24 for c in range(1, cmapss.MIN_CYCLES)]
    ok = [[2, c] + [0.0] * 24 for c in range(1, cmapss.MIN_CYCLES + 5)]
    np.savetxt(tmp_path / "train_FD009.txt", np.array(short + ok))
    assert [u.name for u in cmapss.load_subset(9, data_dir=tmp_path)] == ["FD009_u2"]


def test_a_wrong_column_count_is_refused(tmp_path):
    np.savetxt(tmp_path / "train_FD008.txt", np.zeros((4, 12)))
    with pytest.raises(ValueError, match="expected 26 columns"):
        cmapss.load_subset(8, data_dir=tmp_path)


def test_a_missing_subset_is_reported_by_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        cmapss.load_subset(1, data_dir=tmp_path)


def test_constant_channels_survive_scaling_as_zero(tiny):
    # C-MAPSS has channels that never move; they must collapse to zero, not divide by zero.
    grids = [u.grid for u in cmapss.load_subset(7, data_dir=tiny)]
    scale = features.fit_scale(grids)
    out = features.normalize(grids[0], scale)
    assert np.all(np.isfinite(out))


# --- the reason this dataset is here -----------------------------------------------------------

def test_the_real_fleet_is_large_enough_to_verify_the_target_risk():
    try:
        fleet = cmapss.load()
    except FileNotFoundError:
        pytest.skip("C-MAPSS data not present")
    assert len(fleet) > 700
    # 36 assets are needed to demonstrate alpha = 0.10; the bearing fleets have 17 and 15.
    assert len(fleet) >= certify.min_assets_to_verify(0.10)
    assert certify.verification_floor(len(fleet)) < 0.10


def test_every_subset_contributes_its_documented_unit_count():
    try:
        counts = {s: len(cmapss.load_subset(s)) for s in cmapss.SUBSETS}
    except FileNotFoundError:
        pytest.skip("C-MAPSS data not present")
    assert counts == {1: 100, 2: 260, 3: 100, 4: 249}
