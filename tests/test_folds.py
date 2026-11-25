"""Tests for nested, condition-stratified fold assignment."""
import numpy as np
import pytest

from certalarm import folds


class FakeAsset:
    def __init__(self, name, condition):
        self.name, self.condition = name, condition


def _fleet(per_cond=6, n_cond=3):
    return [FakeAsset(f"Bearing{c}_{i}", c)
            for c in range(1, n_cond + 1) for i in range(1, per_cond + 1)]


def test_outer_test_groups_partition_the_fleet():
    a = _fleet()
    got = [n for f in folds.nested_folds(a, seed=0) for n in f["test"]]
    assert sorted(got) == sorted(x.name for x in a)


def test_calibration_is_exactly_the_complement_of_test():
    a = _fleet()
    allnames = {x.name for x in a}
    for f in folds.nested_folds(a, seed=0):
        assert set(f["calibration"]) == allnames - set(f["test"])


def test_inner_held_sets_partition_the_calibration_pool():
    for f in folds.nested_folds(_fleet(), seed=1):
        held = [n for inner in f["inner"] for n in inner["held"]]
        assert sorted(held) == sorted(f["calibration"])


def test_inner_fit_never_overlaps_inner_held():
    for f in folds.nested_folds(_fleet(), seed=1):
        for inner in f["inner"]:
            assert not set(inner["fit"]) & set(inner["held"])
            assert set(inner["fit"]) | set(inner["held"]) == set(f["calibration"])


def test_no_test_asset_ever_appears_in_any_inner_split():
    for f in folds.nested_folds(_fleet(), seed=2):
        test = set(f["test"])
        for inner in f["inner"]:
            assert not test & set(inner["fit"])
            assert not test & set(inner["held"])


def test_every_outer_fold_holds_out_at_least_one_asset():
    assert all(f["test"] for f in folds.nested_folds(_fleet(per_cond=2), seed=0))


def test_stratification_spreads_conditions_across_outer_folds():
    a = _fleet(per_cond=6, n_cond=3)
    cond = {x.name: x.condition for x in a}
    for f in folds.nested_folds(a, seed=0):
        assert len({cond[n] for n in f["test"]}) >= 2


def test_assignment_is_deterministic_for_a_seed():
    a = _fleet()
    assert folds.nested_folds(a, seed=5) == folds.nested_folds(a, seed=5)


def test_different_seeds_give_different_assignments():
    a = _fleet()
    assert folds.nested_folds(a, seed=5) != folds.nested_folds(a, seed=6)


def test_fewer_assets_than_folds_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        folds.nested_folds(_fleet(per_cond=1, n_cond=1), n_outer=5)


def _uneven_fleet():
    """PRONOSTIA's shape: 7 and 7 assets at two conditions, only 3 at the third."""
    return ([FakeAsset(f"Bearing1_{i}", 1) for i in range(1, 8)]
            + [FakeAsset(f"Bearing2_{i}", 2) for i in range(1, 8)]
            + [FakeAsset(f"Bearing3_{i}", 3) for i in range(1, 4)])


def test_a_condition_too_small_to_reach_every_fold_is_not_pinned_to_the_same_folds():
    a = _uneven_fleet()
    cond = {x.name: x.condition for x in a}
    placements = {tuple(i for i, g in enumerate(folds.nested_folds(a, seed=s))
                        if any(cond[n] == 3 for n in g["test"]))
                  for s in range(12)}
    assert len(placements) > 1


def test_outer_group_sizes_stay_within_one_asset_of_each_other():
    for seed in range(8):
        sizes = [len(g["test"]) for g in folds.nested_folds(_uneven_fleet(), seed=seed)]
        assert max(sizes) - min(sizes) <= 1
