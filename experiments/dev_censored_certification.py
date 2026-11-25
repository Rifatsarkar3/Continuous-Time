"""Certify a monitoring alarm from censored maintenance records. C-MAPSS development screen.

THE IDEA. A unit replaced healthy at observation index c (right-censored: it never failed on
record) still bounds its own lead-point statistic. Failure comes later than c, so the last
acceptable alarm index t* = floor(0.95 (n-1)) is at least floor(0.95 c), and the running minimum
is non-increasing, hence

    theta* = m(t*)  <=  m(floor(0.95 c)) =: s          (observable without knowing n)

Using s as the censored unit's conformal score dominates its true lateness, so split-conformal on
{theta* of failed units} U {s of censored units} keeps its finite-sample validity, provided the
(trajectory, failure time, censoring time) triples are exchangeable. Censored units - which every
prognostics method discards - become calibration data.

DECLARED BEFORE RUNNING (2026-09-23). Nothing below is tuned after the fact.

  Fleet per replicate: N_F = 14 run-to-failure units (the bearing-benchmark regime, where the
  certificate is the maximum) plus N_C = 150 censored units, all drawn from the E4 calibration
  pool of partition seed 42. Held-out validity/efficiency is measured on the 150 E4 test units,
  which are true failures. The 100 reserved units are never read.

  Censoring policies, simulated with known failure times:
    P1 condition-based: removed when true normalized RUL falls to r ~ U(0.10, 0.40)
    P2 time-based early: removed when true normalized RUL falls to r ~ U(0.40, 0.90)

  Arms: A = failed only (n = 14); B = failed + censored (n = 164);
        C = oracle, as if the 150 censored units had been run to failure.

  Criteria:
    validity  - B's mean held-out late rate <= alpha + 2 Monte-Carlo s.e., under both policies
    efficiency- under P1, B's mean held-out waste <= 0.90 x A's   (>= 10% reduction)
  Prediction: P1 passes; P2 does not reach 10%, because early removal yields loose bounds.
  Reported either way.

Run once:  python experiments/dev_censored_certification.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import cmapss, evaluate, features  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from certalarm.predictors import ctgru_train  # noqa: E402
from experiments.e4_asset_count import ALPHA, ITERS, LEAD, make_ctgru, partition  # noqa: E402
from experiments.dev_lead_point_reduction import lead_point  # noqa: E402

SEED, N_F, N_C, REPS = 42, 14, 150, 400
POLICIES = {"P1 condition-based": (0.10, 0.40), "P2 time-based early": (0.40, 0.90)}


def conformal_threshold(scores, thetas, alpha):
    """Smallest grid theta with (#{score >= theta} + 1)/(n+1) <= alpha. Exact CRC, by reduction."""
    s = np.asarray(scores, float)
    n = len(s)
    for th in thetas:
        if (np.sum(s >= th) + 1) / (n + 1) <= alpha + 1e-12:
            return float(th)
    return None


def censored_score(mu, rul, r):
    """Score of a unit removed when true normalized RUL first falls to r. Returns (s, c)."""
    c = int(np.flatnonzero(np.asarray(rul) <= r)[0])      # observed indices 0..c-1
    j = int(np.floor((1 - LEAD) * c))
    j = min(j, c - 1)
    return float(np.min(np.asarray(mu, float)[: j + 1])), c


def main():
    fleet = cmapss.load()
    train, pool, test, reserved = partition(fleet, SEED)
    assert not {u.name for u in reserved} & {u.name for u in train + pool + test}
    scale = features.fit_scale([u.grid for u in train])
    X = [features.normalize(u.grid, scale) for u in train]
    print(f"training CT-GRU on {len(train)} units, {ITERS} iters (seed {SEED}) ...", flush=True)
    predict = make_ctgru(ctgru_train.train(X, [u.rul for u in train], "full", ITERS, SEED), scale)

    thetas = np.asarray(evaluate.THETAS, float)
    P = [(np.asarray(predict(u), float), np.asarray(u.rul, float)) for u in pool]
    tstar_pool = np.array([lead_point(mu, rul, LEAD) for mu, rul in P])
    T_late, T_waste = [], []
    for u in test:
        l, w = outcome_curves(predict(u), np.asarray(u.rul, float), thetas, LEAD,
                              np.ones(u.n_min, bool))
        T_late.append(l)
        T_waste.append(w)
    T_late, T_waste = np.stack(T_late), np.stack(T_waste)
    print(f"pool {len(pool)}  test {len(test)}  (reserved {len(reserved)} untouched)\n")

    def held_out(th):
        k = int(np.searchsorted(thetas, th))
        return T_late[:, k].mean(), T_waste[:, k].mean()

    rng = np.random.default_rng(SEED)
    for pname, (lo, hi) in POLICIES.items():
        res = {a: [] for a in "ABC"}
        bound_ok = 0
        for _ in range(REPS):
            idx = rng.permutation(len(pool))
            f_idx, c_idx = idx[:N_F], idx[N_F:N_F + N_C]
            s_c = []
            for i in c_idx:
                mu, rul = P[i]
                s, _ = censored_score(mu, rul, rng.uniform(lo, hi))
                assert s >= tstar_pool[i] - 1e-12, "censored bound violated"
                bound_ok += 1
                s_c.append(s)
            th = {"A": conformal_threshold(tstar_pool[f_idx], thetas, ALPHA),
                  "B": conformal_threshold(np.r_[tstar_pool[f_idx], s_c], thetas, ALPHA),
                  "C": conformal_threshold(tstar_pool[np.r_[f_idx, c_idx]], thetas, ALPHA)}
            for a in "ABC":
                res[a].append((th[a], *held_out(th[a])))
        print(f"== {pname}: censored at true RUL ~ U({lo}, {hi})  "
              f"[bound s >= theta* held on all {bound_ok} censored draws]")
        print(f"   {'arm':<34}{'theta':>7}{'late':>8}{'+-2se':>7}{'waste':>8}")
        lab = {"A": f"A  failed only (n={N_F})", "B": f"B  failed + censored (n={N_F + N_C})",
               "C": f"C  oracle: all {N_F + N_C} run to failure"}
        for a in "ABC":
            r = np.array(res[a])
            se = r[:, 1].std(ddof=1) / np.sqrt(len(r))
            print(f"   {lab[a]:<34}{r[:, 0].mean():>7.3f}{r[:, 1].mean():>8.3f}"
                  f"{2 * se:>7.3f}{r[:, 2].mean():>8.3f}")
        rB, rA = np.array(res["B"]), np.array(res["A"])
        se_b = rB[:, 1].std(ddof=1) / np.sqrt(len(rB))
        valid = rB[:, 1].mean() <= ALPHA + 2 * se_b
        gain = (rB[:, 2].mean() - rA[:, 2].mean()) / rA[:, 2].mean()
        print(f"   validity (B late <= alpha + 2se): {'PASS' if valid else 'FAIL'}")
        print(f"   efficiency B vs A: {gain:+.1%}   criterion <= -10%: "
              f"{'PASS' if gain <= -0.10 else 'FAIL'}\n")


if __name__ == "__main__":
    main()
