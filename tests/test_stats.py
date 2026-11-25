"""Tests for the gate statistics."""
import numpy as np
import pytest

from certalarm import stats


def test_clopper_pearson_brackets_the_point_estimate():
    lo, hi = stats.clopper_pearson(3, 20)
    assert lo < 3 / 20 < hi


def test_clopper_pearson_endpoints_are_exact_at_zero_and_full():
    assert stats.clopper_pearson(0, 10)[0] == 0.0
    assert stats.clopper_pearson(10, 10)[1] == 1.0


def test_clopper_pearson_matches_a_known_value():
    lo, hi = stats.clopper_pearson(2, 20, alpha=0.05)
    assert lo == pytest.approx(0.0123, abs=5e-4)
    assert hi == pytest.approx(0.3170, abs=5e-4)


def test_clopper_pearson_interval_narrows_with_more_trials():
    narrow = stats.clopper_pearson(10, 100)
    wide = stats.clopper_pearson(1, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_clopper_pearson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        stats.clopper_pearson(11, 10)


def test_holm_preserves_order_and_inflates_small_p_values():
    raw = [0.01, 0.04, 0.03]
    adj = stats.holm(raw)
    assert len(adj) == 3
    assert all(a >= r for a, r in zip(adj, raw))
    assert adj[0] == pytest.approx(0.03)


def test_holm_is_monotone_in_the_sorted_order():
    adj = stats.holm([0.001, 0.002, 0.3, 0.4])
    order = np.argsort([0.001, 0.002, 0.3, 0.4])
    assert np.all(np.diff(np.asarray(adj)[order]) >= -1e-12)


def test_paired_wilcoxon_detects_a_consistent_improvement():
    a = np.arange(1.0, 13.0)
    res = stats.paired_wilcoxon(a, a - 0.5)
    assert res["median_delta"] == pytest.approx(0.5)
    assert res["p"] < 0.05
    assert res["n"] == 12


def test_paired_wilcoxon_reports_a_tie_honestly():
    a = np.arange(1.0, 13.0)
    res = stats.paired_wilcoxon(a, a.copy())
    assert res["median_delta"] == pytest.approx(0.0)
    assert np.isnan(res["p"])


def test_paired_wilcoxon_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        stats.paired_wilcoxon([1.0, 2.0], [1.0])


def test_cluster_bootstrap_resamples_assets_not_rows():
    # Two assets; every row of an asset is identical, so any asset-level resample
    # can only ever produce means of 0.0, 0.5 or 1.0.
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    asset = np.array(["a", "a", "a", "b", "b", "b"])
    boot = stats.cluster_bootstrap_mean(values, asset, n_boot=200, seed=0)
    assert set(np.round(boot, 6)) <= {0.0, 0.5, 1.0}


def test_cluster_bootstrap_interval_contains_the_observed_mean():
    rng = np.random.default_rng(0)
    values = rng.random(60)
    asset = np.repeat([f"b{i}" for i in range(12)], 5)
    lo, hi = stats.cluster_bootstrap_ci(values, asset, n_boot=500, seed=1)
    assert lo <= values.mean() <= hi


def test_holm_keeps_an_untested_comparison_out_of_the_family():
    # A NaN means "no test happened". It must not become a finite p-value, and it must not
    # consume a rank and inflate the genuine comparisons beside it.
    adj = stats.holm([0.001, float("nan"), 0.20])
    assert np.isnan(adj[1])
    assert adj[0] == pytest.approx(0.002)
    assert adj[2] == pytest.approx(0.20)


def test_holm_of_all_untested_comparisons_is_all_nan():
    assert all(np.isnan(x) for x in stats.holm([float("nan"), float("nan")]))


def test_paired_wilcoxon_reports_how_many_pairs_actually_counted():
    a = np.arange(1.0, 13.0)
    b = a.copy()
    b[:4] -= 0.5                      # only 4 of 12 pairs differ
    res = stats.paired_wilcoxon(a, b)
    assert res["n"] == 12
    assert res["n_effective"] == 4


def test_clopper_pearson_refuses_a_fractional_count():
    with pytest.raises(ValueError, match="whole counts"):
        stats.clopper_pearson(0.75, 4)


def test_cluster_bootstrap_weights_every_asset_equally():
    # asset "a" contributes 20 rows, asset "b" only 2; an asset-level claim must not be
    # dominated by the row-heavy asset.
    values = np.array([0.0] * 20 + [1.0] * 2)
    asset = np.array(["a"] * 20 + ["b"] * 2)
    assert stats.cluster_bootstrap_mean(values, asset, n_boot=500, seed=0).mean() == \
        pytest.approx(0.5, abs=0.05)
