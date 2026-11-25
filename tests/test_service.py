import itertools
import math

from math import comb

import numpy as np
import pytest

from certalarm import service as S


def fleet(rng, n=40, length=60, tau=3.0):
    """Noisy, non-monotone signals with random lives: a late alarm can happen at any point of life."""
    out = []
    for i in range(n):
        T = int(rng.integers(length // 2, length))
        s = np.linspace(1.0, 0.0, T + 1) + rng.normal(0, 0.3, T + 1)
        out.append(S.Asset(f"a{i}", 0, np.arange(T + 1.0), s, float(T), tau))
    return out


def test_late_count_matches_enumeration():
    n, m, k = 3, 4, 1
    counts = np.zeros(m + 1)
    for cal in itertools.combinations(range(n + m), n):              # ranks, 0 = largest
        edge = sorted(cal)[k]                                        # (k+1)-th largest calibration value
        counts[sum(r < edge for r in range(n + m) if r not in cal)] += 1
    assert S.late_count(n, m, k).pmf(np.arange(m + 1)) == pytest.approx(counts / counts.sum())


def test_zero_failure_odds_are_n_over_n_plus_m():
    for n, m in ((9, 36), (18, 36), (36, 36), (32, 36)):
        assert S.pass_odds(n, m) == pytest.approx(n / (n + m))
    assert S.pass_odds(22, 20, c=3) == pytest.approx(1 - math.comb(20, 4) / math.comb(42, 4))


def test_demonstration_sizes():
    assert [S.demo_size(0.10, d) for d in (0.10, 0.05, 0.025)] == [22, 29, 36]
    for c in (0, 1, 5):
        m = S.demo_size(0.10, 0.025, c)
        assert S.binom_cdf(c, m, 0.10) <= 0.025 < S.binom_cdf(c, m - 1, 0.10)


def test_ceiling_of_the_level_rule():
    for alpha, delta in ((0.10, 0.025), (0.10, 0.05), (0.05, 0.025), (0.20, 0.025), (0.02, 0.10)):
        m = S.demo_size(alpha, delta)
        ns = range(math.ceil(1 / alpha) - 1, int(40 / alpha))
        odds = [S.pass_odds(n, m, k=S.level_rank(n, alpha)) for n in ns]
        best = list(ns)[int(np.argmax(odds))]
        assert best == math.ceil(round(2 / alpha, 9)) - 2
        assert max(odds) == pytest.approx(best / (best + m))
    assert 18 / (18 + 36) == pytest.approx(1 / 3)


def test_smallest_fleet_is_smallest():
    N, n, m, c = S.smallest_fleet(0.10, 0.025, 0.80)
    assert S.pass_odds(n, m, c) >= 0.80 and N == n + m and m == S.demo_size(0.10, 0.025, c)
    for c2 in range(8):
        m2 = S.demo_size(0.10, 0.025, c2)
        assert all(S.pass_odds(n2, m2, c2) < 0.80 for n2 in range(1, N - m2))


def test_pooling_identity():
    rng = np.random.default_rng(0)
    for k in (0, 1, 3):
        for _ in range(50):
            v = rng.normal(size=30)
            cal, test = v[:12], v[12:]
            x = int((test >= S.threshold(cal, k)).sum())
            assert S.threshold(cal, k) == S.threshold(v, k + x)


def test_in_service_threshold_equals_complete_records():
    rng = np.random.default_rng(1)
    for k in (0, 1, 3):
        f = fleet(rng)
        runs = S.run_in_service(f, k)
        values = [a.lead_value for a in f]
        for t, r in enumerate(runs):
            full = S.threshold(values[:t], k) if t > k else -np.inf
            assert r["theta"] == full
            assert r["late"] == (values[t] >= full)
        assert 0 < sum(r["late"] for r in runs) < len(f)


def test_in_service_never_reads_a_replaced_assets_future():
    """Scrambling everything after the replacement of an on-time asset cannot change any later threshold."""
    rng = np.random.default_rng(2)
    f = fleet(rng)
    base = S.run_in_service(f, 1)
    g = []
    for a, r in zip(f, base):
        s = a.s.copy()
        if not r["late"]:
            s[r["index"] + 1:] = rng.normal(0, 5, len(s) - r["index"] - 1)
        g.append(S.Asset(a.name, a.group, a.t, s, a.T, a.tau))
    assert [r["theta"] for r in S.run_in_service(g, 1)] == [r["theta"] for r in base]


def test_record_law_enumeration_and_mean():
    N = 6
    counts = np.zeros(N + 1)
    for p in itertools.permutations(range(N)):
        counts[sum(p[t] > max(p[:t], default=-1) for t in range(N))] += 1
    assert S.count_pmf(S.record_probs(N)) == pytest.approx(counts / counts.sum())
    for k in (0, 2):
        H = lambda n: sum(1 / i for i in range(1, n + 1))
        assert S.record_probs(50, k).sum() == pytest.approx((k + 1) * (1 + H(50) - H(k + 1)))


def test_record_law_by_simulation():
    rng = np.random.default_rng(3)
    N, k = 30, 1
    got = [sum(r["late"] for r in S.run_in_service(
        [S.Asset("x", 0, np.zeros(1), np.array([v]), 1.0, 0.0) for v in rng.normal(size=N)], k)) for _ in range(400)]
    p = S.record_probs(N, k)
    assert abs(np.mean(got) - p.sum()) < 4 * math.sqrt((p * (1 - p)).sum() / 400)


def test_order_weights_match_enumeration():
    N, n, k = 7, 4, 1
    w = np.zeros(N)
    for c in itertools.combinations(range(N), n):
        w[sorted(c)[k]] += 1
    assert S.order_weights(N, n, k) == pytest.approx(w / w.sum())


def test_level_rule_with_bounds_is_conservative():
    rng = np.random.default_rng(4)
    for _ in range(10):
        f = fleet(rng, n=60)
        values = [a.lead_value for a in f]
        for t, r in enumerate(S.run_level_in_service(f, 0.10)):
            if t:
                full = S.threshold(values[:t], max(0, S.level_rank(t, 0.10)))
                assert r["theta"] >= full
                assert r["late"] <= (values[t] >= full)


def test_short_asset_is_always_late():
    a = S.Asset("x", 0, np.arange(3.0), np.array([1.0, 0.5, 0.2]), 2.0, 5.0)
    assert a.lead_value == np.inf
    assert S.threshold([np.inf, 0.1], 0) == np.inf


def test_two_fixed_cost_laws():
    M = 100
    cost = [n + (M - n) / (n + 1) for n in range(1, M)]
    assert int(np.argmin(cost)) + 1 == round(math.sqrt(M + 1) - 1)
    assert S.record_probs(M).sum() < min(cost) / 3


def test_each_regime_costs_at_most_its_own_records():
    """Assets exchangeable within regimes, arbitrary between them: late in service implies a record of its own regime."""
    rng = np.random.default_rng(5)
    sizes, shifts, k = (15, 20, 10), (0.0, 5.0, -3.0), 1
    total = []
    for _ in range(300):
        v = np.concatenate([rng.normal(m, 1, n) for m, n in zip(shifts, sizes)])
        late = [r["late"] for r in S.run_in_service(
            [S.Asset("x", 0, np.zeros(1), np.array([x]), 1.0, 0.0) for x in v], k)]
        total.append(sum(late))
        start = 0
        for n in sizes:
            for i in range(n):
                own = sorted(v[start:start + i], reverse=True)
                if late[start + i] and len(own) > k:
                    assert v[start + i] > own[k]
            start += n
    assert np.mean(total) <= sum(S.record_probs(n, k).sum() for n in sizes)


def test_overlapping_service_is_bounded_whatever_the_dependence_on_life():
    """Lives tied to the lead-point value, so that the assets completed first are a selected sample."""
    rng = np.random.default_rng(6)
    for k, slots, sign in ((0, 4, 1), (0, 4, -1), (1, 6, -1), (0, 1, 1)):
        late_sum, N, reps = np.zeros(40), 40, 300
        for _ in range(reps):
            v = rng.normal(size=N)
            f = [S.Asset(f"a{i}", 0, np.zeros(1), np.array([x]), float(np.exp(sign * 2 * x) + 1e-6), 0.0) for i, x in enumerate(v)]
            runs = S.run_overlapping(f, slots, k)
            for j, r in enumerate(runs):
                late_sum[j] += r["late"]
                if r["late"]:                                          # rank among the first j + 1 entrants
                    assert (v[: j + 1] > v[j]).sum() < k + slots
            if slots == 1:
                assert [r["late"] for r in runs] == [r["late"] for r in S.run_in_service(f, k)]
        bound = np.minimum(1.0, (k + slots) / np.arange(1, N + 1))
        assert (late_sum / reps <= bound + 4 * np.sqrt(bound * (1 - bound) / reps) + 1e-9).all()


def test_overlapping_threshold_equals_complete_records_of_the_completed():
    rng = np.random.default_rng(7)
    f = fleet(rng, n=50)
    for k in (0, 2):
        runs = S.run_overlapping(f, 5, k)
        # rebuild the completed set at each entry from the outcomes and compare with the full records
        end, start = [], []
        import heapq as hq
        busy, clock = [], 0.0
        for j, (a, r) in enumerate(zip(f, runs)):
            if len(busy) == 5:
                clock = hq.heappop(busy)
                while busy and busy[0] <= clock:
                    hq.heappop(busy)
            else:
                while busy and busy[0] <= clock:
                    hq.heappop(busy)
            start.append(clock)
            end.append(clock + r["time"])
            hq.heappush(busy, end[-1])
            done = [i for i in range(j) if end[i] <= clock]
            assert len(done) == r["completed"]
            if len(done) > k:
                assert r["theta"] == S.threshold([f[i].lead_value for i in done], k)


def test_late_tail_is_exact():
    for n, m, k in ((9, 36, 0), (32, 36, 1), (20, 15, 3)):
        for x in range(m + 1):
            assert S.late_tail(n, m, x, k) == pytest.approx(S.late_count(n, m, k).sf(x - 1), abs=1e-12)
    assert S.late_tail(41, 43, 40) == pytest.approx(comb(43, 40) / comb(84, 40), rel=1e-12)     # 8e-21, below scipy's floor
