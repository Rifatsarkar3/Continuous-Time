"""Tests for turning trajectories into certified thresholds and outcomes."""
import numpy as np
import pytest

from certalarm import evaluate
from certalarm.crossfit import Trajectory


def _traj(n=50, bias=0.0, noise=0.0, seed=0, observed=None):
    rul = np.linspace(1.0, 0.0, n)
    rng = np.random.default_rng(seed)
    mu = np.clip(rul + bias + noise * rng.standard_normal(n), 0, 1)
    can = np.ones(n, bool) if observed is None else observed
    return Trajectory(mu, np.full(n, 0.05), rul, can)


def _fleet(k=12, bias=0.0, seed=0):
    return {f"B{i}": [_traj(bias=bias, seed=seed + i)] for i in range(k)}


def test_loss_matrix_is_non_increasing_in_theta():
    L = evaluate.loss_matrix(_fleet(), evaluate.THETAS, lead=0.05)
    assert L.shape == (12, len(evaluate.THETAS))
    assert np.all(np.diff(L, axis=1) <= 1e-12)


def test_loss_matrix_rows_are_in_the_unit_interval():
    L = evaluate.loss_matrix(_fleet(), evaluate.THETAS, lead=0.05)
    assert L.min() >= 0.0 and L.max() <= 1.0


def test_a_perfect_predictor_is_never_late_at_a_generous_threshold():
    L = evaluate.loss_matrix(_fleet(), evaluate.THETAS, lead=0.05)
    assert L[:, -1].max() == 0.0


def test_crc_threshold_certifies_when_there_are_enough_assets():
    out = evaluate.fit_rules(_fleet(k=12), alpha=0.10, lead=0.05)
    assert out["crc"]["theta"] is not None
    assert out["crc"]["certifiable"] is True


def test_crc_refuses_below_the_certifiability_floor():
    out = evaluate.fit_rules(_fleet(k=4), alpha=0.10, lead=0.05)
    assert out["crc"]["theta"] is None
    assert out["crc"]["certifiable"] is False


def test_crc_threshold_is_at_least_the_uncorrected_one():
    cal = _fleet(k=12, bias=-0.05, seed=3)
    out = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    assert out["crc"]["theta"] >= out["empirical"]["theta"]


def test_every_rule_is_reported():
    out = evaluate.fit_rules(_fleet(k=12), alpha=0.10, lead=0.05)
    assert set(out) == set(evaluate.RULES)


def test_apply_rules_reports_late_and_waste_per_asset():
    cal = _fleet(k=12)
    rules = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, _fleet(k=3, seed=99), lead=0.05)
    assert set(got) == set(evaluate.RULES)
    for per_asset in got.values():
        assert len(per_asset) == 3
        for row in per_asset.values():
            assert 0.0 <= row["late"] <= 1.0
            assert 0.0 <= row["waste"] <= 1.0


def test_an_uncertifiable_rule_is_recorded_not_silently_replaced():
    rules = evaluate.fit_rules(_fleet(k=4), alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, _fleet(k=2, seed=5), lead=0.05)
    for row in got["crc"].values():
        assert row["late"] == 1.0
        assert row["waste"] == 0.0


def test_an_optimistic_predictor_alarms_late():
    cal = _fleet(k=12, bias=0.30, seed=11)
    rules = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, {"T": [_traj(bias=0.30, seed=77)]}, lead=0.05)
    assert got["point"]["T"]["late"] == 1.0


def test_unobserved_minutes_cannot_raise_an_alarm():
    obs = np.zeros(50, bool)
    obs[:10] = True
    late_only = {"T": [_traj(observed=obs)]}
    rules = evaluate.fit_rules(_fleet(k=12), alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, late_only, lead=0.05)
    assert got["crc"]["T"]["late"] == 1.0


def test_summarize_pools_assets_and_returns_an_upper_bound():
    cal = _fleet(k=12)
    rules = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, _fleet(k=6, seed=31), lead=0.05)
    s = evaluate.summarize(got["crc"])
    assert s["n_assets"] == 6
    assert 0.0 <= s["late_rate"] <= 1.0
    assert s["late_upper"] >= s["late_rate"]


def test_summarize_reports_the_more_conservative_of_the_two_bounds():
    cal = _fleet(k=12)
    rules = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, _fleet(k=6, seed=31), lead=0.05)
    s = evaluate.summarize(got["crc"])
    assert s["late_upper"] == max(s["late_upper_bootstrap"], s["late_upper_binomial"])


def test_summarize_accepts_fractional_late_rates():
    # A replicate-averaged outcome is fractional; it must not be truncated to a whole count.
    per_asset = {f"B{i}": {"late": 0.375, "waste": 0.2} for i in range(6)}
    s = evaluate.summarize(per_asset)
    assert s["late_rate"] == pytest.approx(0.375)


def _noisy_fleet(k=12, n_rep=4, seed=0):
    return {f"B{i}": [_traj(noise=0.05, seed=1000 * i + r) for r in range(n_rep)]
            for i in range(k)}


def test_supband_averages_replicates_rather_than_taking_their_worst_case():
    # Maximising over replicates as well as over time would inflate the band, making this
    # baseline waste more life and flattering the rule it is compared against.
    cal = _noisy_fleet()
    got = evaluate._supband_shift(cal, alpha=0.10)
    worst_case = evaluate.baselines.supband_quantile(
        [max(evaluate.baselines.sup_score(t.mu, t.sigma, t.rul, t.can_fire) for t in reps)
         for reps in cal.values()], 0.10)
    assert got < worst_case


def test_supband_shift_is_finite_and_usable():
    shift = evaluate._supband_shift(_noisy_fleet(), alpha=0.10)
    assert np.isfinite(shift)


def test_summarize_clusters_rows_by_asset_not_by_row_key():
    # Five seeds of one bearing are one asset, not five independent observations.
    rows, key = {}, {}
    for bearing in ("B0", "B1", "B2", "B3", "B4", "B5"):
        for seed in range(5):
            k = f"{seed}/{bearing}"
            rows[k] = {"late": 1.0 if bearing == "B0" else 0.0, "waste": 0.2}
            key[k] = bearing
    s = evaluate.summarize(rows, cluster_key=key)
    assert s["n_assets"] == 6
    assert s["n_rows"] == 30
    assert s["late_rate"] == pytest.approx(1 / 6)


def test_clustering_by_asset_gives_a_wider_bound_than_treating_rows_as_independent():
    rows, key = {}, {}
    for bearing in ("B0", "B1", "B2", "B3", "B4", "B5"):
        for seed in range(5):
            k = f"{seed}/{bearing}"
            rows[k] = {"late": 1.0 if bearing == "B0" else 0.0, "waste": 0.2}
            key[k] = bearing
    clustered = evaluate.summarize(rows, cluster_key=key)["late_upper"]
    naive = evaluate.summarize(rows)["late_upper"]
    assert clustered > naive


def test_pooling_across_observation_processes_keeps_every_row():
    """The report tables key rows into a dict, so the key must name every distinguishing field.

    A key of seed/fold/asset collides each row with its twin from the other observation
    processes and silently keeps only the last one, discarding two thirds of the evidence.
    """
    import experiments.e2_main_alarms as e2
    import experiments.report_progress as rp

    rows = [{"process": p, "seed": 42, "fold": 0, "asset": "Bearing1_1", "rule": "crc",
             "late": late, "waste": 0.5, "certifiable": True, "predictor": "ctgru",
             "predictor_mae": 0.1}
            for p, late in (("full", 0.0), ("mixture", 0.5), ("sparse_shift", 1.0))]

    for module in (e2, rp):
        keyed = {}
        for r in rows:
            keyed[f"{r['process']}/{r['seed']}/{r['fold']}/{r['asset']}"] = r
        assert len(keyed) == 3, f"{module.__name__} must not collide rows across processes"

    # The pooled late rate is the mean over all three, not whichever process sorted last.
    keyed = {f"{r['process']}/{r['seed']}/{r['fold']}/{r['asset']}": r for r in rows}
    s = evaluate.summarize(keyed, cluster_key={k: "Bearing1_1" for k in keyed})
    assert s["n_rows"] == 3
    assert s["late_rate"] == pytest.approx(0.5)


def test_condition_scaled_rule_scores_test_assets_on_the_scaled_signal():
    # crc_scaled is calibrated on mu / s_k; scoring test assets on raw mu silently replaced the
    # pre-registered rule mu < theta * s_k (found 2026-11-18).
    cal, test = _fleet(k=12, bias=0.2), _fleet(k=3, bias=0.2, seed=99)
    for reps in list(cal.values()) + list(test.values()):
        for t in reps:
            t.scale = 2.0
    rules = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    got = evaluate.apply_rules(rules, test, lead=0.05)["crc_scaled"]
    theta = np.array([rules["crc_scaled"]["theta"]])
    for name, reps in test.items():
        late, rem = evaluate._asset_curves(reps, theta, 0.05, evaluate._scaled_signal)
        assert got[name] == {"late": float(late[0]), "waste": float(rem[0])}


def test_condition_scaled_rule_certifies_when_scaled_signal_exceeds_one():
    # A scale below 1 lifts mu / s above 1; the certificate must still exist - a grid
    # that stops at 1.0 made such folds "decline" (found in review, 2026-11-19).
    cal = _fleet(k=12, bias=0.6)        # lead-point mu 0.65, scaled 1.3 > 1
    for reps in cal.values():
        for t in reps:
            t.scale = 0.5
    rules = evaluate.fit_rules(cal, alpha=0.10, lead=0.05)
    assert rules["crc_scaled"]["certifiable"] is True
    assert rules["crc_scaled"]["theta"] > 1.0


def test_clopper_pearson_bound_is_valid_for_fractional_late_rates():
    # Every asset late in 40% of its replicates: rounding to 0/1 gave zero late assets and a bound far
    # below the true mean risk 0.4; the bound must cover it (found in review, 2026-11-19).
    per_asset = {f"A{i}": {"late": 0.4, "waste": 0.5} for i in range(15)}
    s = evaluate.summarize(per_asset)
    assert s["late_rate"] == pytest.approx(0.4)
    assert s["late_upper_binomial"] >= 0.4
