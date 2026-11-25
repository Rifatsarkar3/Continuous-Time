"""Tests for the pre-registered C1 analysis: permutation p, paired summary, fixed sequence."""
import numpy as np
import pytest

from experiments import c1_leadpoint_xjtu as c1


def test_signflip_p_is_exact_for_a_uniform_positive_shift():
    assert c1.signflip_p(np.full(5, 0.3)) == pytest.approx(2 / 32)


def test_signflip_p_is_one_when_there_is_no_difference():
    assert c1.signflip_p(np.zeros(6)) == 1.0


def test_paired_summary_interval_brackets_the_mean_and_counts_directions():
    d = np.array([-0.2, -0.1, 0.0, 0.1, 0.3, 0.05, -0.05])
    s = c1.paired_summary(d)
    assert s["ci95"][0] < s["mean"] < s["ci95"][1]
    assert s["mean"] < s["upper_one_sided_95"] < s["ci95"][1]
    assert (s["lp_lower"], s["lp_higher"]) == (3, 3)


def _s(upper):
    return {"upper_one_sided_95": upper}


def test_fixed_sequence_stops_at_the_first_failure():
    seq = c1.fixed_sequence(_s(0.08), _s(-0.1))
    assert [x["result"] for x in seq] == ["FAIL", "NOT TESTED", "NOT TESTED"]
    seq = c1.fixed_sequence(_s(0.02), _s(0.01))
    assert [x["result"] for x in seq] == ["PASS", "FAIL", "NOT TESTED"]
    seq = c1.fixed_sequence(_s(-0.01), _s(-0.02))
    assert [x["result"] for x in seq] == ["PASS", "PASS", "PASS"]


def _row(asset, seed, waste=0.5):
    return {"predictor": "p", "rule": "crc", "process": "full", "seed": seed, "asset": asset,
            "waste": waste, "late": 0.0, "certifiable": True, "predictor_mae": 0.2}


def test_per_bearing_averages_over_the_three_seeds():
    rows = [_row("B1", s, w) for s, w in zip(c1.SEEDS, (0.3, 0.6, 0.9))]
    assert c1.per_bearing(rows, "p")["B1"]["waste"] == pytest.approx(0.6)


def test_per_bearing_refuses_a_bearing_missing_a_seed():
    with pytest.raises(SystemExit):
        c1.per_bearing([_row("B1", 42), _row("B1", 123)], "p")
