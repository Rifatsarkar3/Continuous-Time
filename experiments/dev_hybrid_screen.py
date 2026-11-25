"""Screen: does a classical vibration alarm catch the bearings that set the ML certificate?

PRONOSTIA development trajectories only. Outer test bearings asserted absent; registry untouched.

DECLARED BEFORE RUNNING (2026-09-23). Nothing below is tuned.

  Classical alarm: RMS per minute; baseline mean and sd over the first BASE_MIN minutes;
  fires at the first minute with RMS > mean + K*sd sustained for PERSIST consecutive minutes.
  K = 3, BASE_MIN = 5, PERSIST = 3 - fixed a priori, the three-sigma fault-alarm-point rule
  common in the bearing literature. It is NOT a function of theta.

  Hybrid alarm: fires at the earlier of the ML running-minimum crossing of theta and the
  classical alarm. By the lead-point reduction, an asset is late at theta iff the classical
  alarm misses its last acceptable minute t* AND theta <= theta*. So the hybrid certificate is
  split-conformal over theta* restricted to the assets the classical alarm misses.

  Viability criterion: the hybrid lowers mean wasted life by >= 10% relative to ML-only, at the
  same certification rule. Reported either way.

In-sample within each fold (threshold fitted and scored on the same calibration bearings), so
this is a screen for whether the effect exists, not an estimate of its size.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, certify, crossfit, evaluate  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor  # noqa: E402
from experiments.dev_lead_point_reduction import lead_point  # noqa: E402

SEED, K, BASE_MIN, PERSIST = 42, 3.0, 5, 3


def rms_per_minute(a):
    w = np.asarray(a.windows, float)
    r = np.sqrt((w ** 2).mean(axis=1))
    return r[: a.n_min * a.per_min].reshape(a.n_min, a.per_min).mean(axis=1)


def classical_alarm(rms):
    """First minute of a PERSIST-long run above mean + K*sd of the first BASE_MIN minutes."""
    base = rms[:BASE_MIN]
    thr = base.mean() + K * base.std(ddof=1)
    hot = rms > thr
    hot[:BASE_MIN] = False
    run = 0
    for i, h in enumerate(hot):
        run = run + 1 if h else 0
        if run >= PERSIST:
            return i
    return None


def hybrid_outcome(mu, rul, theta, c_idx):
    """(late, remaining life at alarm) for the earlier of ML crossing and classical alarm."""
    m = np.minimum.accumulate(np.asarray(mu, float))
    hit = np.flatnonzero(m < theta)
    k = hit[0] if hit.size else None
    if c_idx is not None:
        k = c_idx if k is None else min(k, c_idx)
    if k is None:
        return 1.0, 0.0
    rem = float(rul[k])
    return float(rem < LEAD), rem


def main():
    fleet = assets.load("pronostia")
    by = {a.name: a for a in fleet}
    pred = CTGRUPredictor("pronostia")
    thetas = np.asarray(evaluate.THETAS, float)
    rows = []
    for fi, spec in enumerate(crossfit.build_fold(fleet, seed=SEED)):
        own_test = set(spec["test"])
        cal = []
        for j, inner in enumerate(spec["inner"]):
            assert not set(inner["held"]) & own_test and not set(inner["fit"]) & own_test
            model = pred.fit([by[n] for n in inner["fit"]], seed=SEED * 100 + j, process="full")
            for name in inner["held"]:
                a = by[name]
                mu, _ = pred.predict(model, a, np.ones(a.n_min, dtype=bool))
                mu, rul = np.asarray(mu, float), np.asarray(a.rul, float)
                ok = np.flatnonzero(rul >= LEAD)
                tstar = int(ok[-1]) if ok.size else -1
                c = classical_alarm(rms_per_minute(a))
                cal.append({"name": name, "mu": mu, "rul": rul, "ts": lead_point(mu, rul, LEAD),
                            "c": c, "c_ok": c is not None and c <= tstar,
                            "c_rem": float(rul[c]) if c is not None else None})
        # ML-only certificate
        L = np.stack([outcome_curves(x["mu"], x["rul"], thetas, LEAD,
                                     np.ones(len(x["mu"]), bool))[0] for x in cal])
        th_ml = certify.crc_threshold(L, thetas, ALPHA)
        # Hybrid certificate: exact via the reduction
        Lh = np.stack([np.zeros(len(thetas)) if x["c_ok"] else (thetas <= x["ts"]).astype(float)
                       for x in cal])
        th_hy = certify.crc_threshold(Lh, thetas, ALPHA)
        # sanity: the reduction-based hybrid lateness must match direct simulation
        for x, lrow in zip(cal, Lh):
            for ti in (0, len(thetas) // 2, len(thetas) - 1):
                assert hybrid_outcome(x["mu"], x["rul"], thetas[ti], x["c"])[0] == lrow[ti]
        w_ml = np.mean([hybrid_outcome(x["mu"], x["rul"], th_ml, None)[1] for x in cal])
        w_hy = np.mean([hybrid_outcome(x["mu"], x["rul"], th_hy, x["c"])[1] for x in cal])
        l_hy = np.mean([hybrid_outcome(x["mu"], x["rul"], th_hy, x["c"])[0] for x in cal])
        worst = max(cal, key=lambda x: x["ts"])
        rows.append((fi, len(cal), th_ml, th_hy, w_ml, w_hy, l_hy,
                     sum(x["c_ok"] for x in cal), worst["name"], worst["c_ok"]))

    print(f"{'fold':>4} {'n':>3} {'theta ML':>9} {'theta hyb':>10} {'waste ML':>9} "
          f"{'waste hyb':>10} {'late hyb':>9} {'caught':>7}  hostage (caught?)")
    for r in rows:
        print(f"{r[0]:>4} {r[1]:>3} {r[2]:>9.3f} {r[3]:>10.3f} {r[4]:>9.3f} {r[5]:>10.3f} "
              f"{r[6]:>9.3f} {r[7]:>4}/{r[1]:<2}  {r[8]} ({'yes' if r[9] else 'no'})")
    wm, wh = np.mean([r[4] for r in rows]), np.mean([r[5] for r in rows])
    rel = (wh - wm) / wm
    print(f"\nmean waste  ML-only {wm:.3f}   hybrid {wh:.3f}   change {rel:+.1%}")
    print(f"declared criterion (<= -10%): {'VIABLE' if rel <= -0.10 else 'NOT VIABLE'}")


if __name__ == "__main__":
    main()
