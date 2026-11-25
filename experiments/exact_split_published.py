"""Exact split conformal risk control on PRONOSTIA with the authors' published four-bearing predictor.

Descriptive. Pre-declared on 2026-11-19
(commit 1df9160) before any outcome was computed.

The published Dual-Stream network (outputs/r1_final_B10/final20; trusted local files) was
trained and selected on Bearing1_1, 1_2, 2_1 and 2_2 only, so its predictions on the other thirteen bearings
make split CRC exact. Leave-one-out over the ten Condition 1/2 bearings: calibrate on nine (k_n = 0), test
the tenth, for each of the 20 seeds; the certified age rule on the same folds is the paired comparison.
Condition 3 is a descriptive shift check (calibrate on all ten, score the three Condition 3 bearings).

    python experiments/exact_split_published.py      # writes results/exact_split_published.json
"""
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, certify, evaluate, paths, stats  # noqa: E402
from certalarm.predictors.dsaf_train import to_minutes  # noqa: E402
from experiments.age_baseline import age_outcome, certified_age  # noqa: E402
from experiments.c1_leadpoint_xjtu import paired_summary  # noqa: E402

ALPHA, LEAD = 0.10, 0.05
PRED = paths.DUAL_STREAM / "outputs" / "r1_final_B10" / "final20"
TRAIN = {"Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2"}


def trajectories(f, rul):
    """-> {bearing: replicate} on this study's minute grid, Monte Carlo mean as the signal."""
    d = np.load(f)
    out = {}
    for k in d.files:
        if k.startswith("mcmean_"):
            name = "Bearing" + k.split("Bearing")[1]
            mu, _ = to_minutes(d[k], d[k.replace("mcmean_", "mcstd_")], assets.PER_MIN["pronostia"])
            assert len(mu) == len(rul[name]), (name, len(mu), len(rul[name]))
            out[name] = SimpleNamespace(mu=mu, rul=rul[name], can_fire=None)
    return out


def lead_value(t):
    """theta*: running minimum of the signal up to the last on-time minute, the lead-point value."""
    return float(t.mu[:np.flatnonzero(t.rul >= LEAD)[-1] + 1].min())


def grid(cal):
    """This study's grid, extended above the largest calibration signal as evaluate.fit_rules does."""
    ts = evaluate.THETAS
    top = max(float(t.mu.max()) for t in cal.values())
    if top >= ts[-1]:
        ts = np.concatenate([ts, np.arange(ts[-1] + 0.005, np.ceil(top / 0.005) * 0.005 + 0.0101, 0.005)])
    return ts


def certify_on(cal):
    ts = grid(cal)
    th = certify.crc_threshold(evaluate.loss_matrix({a: [t] for a, t in cal.items()}, ts, LEAD), ts, ALPHA)
    # E1: at k_n = 0 the threshold is the first grid point above the largest calibration lead-point value
    top = max(lead_value(t) for t in cal.values())
    assert th == ts[np.searchsorted(ts, top, side="right")], (th, top)
    return th, max(cal, key=lambda a: lead_value(cal[a]))


def outcome(t, th):
    late, rem = evaluate._asset_curves([t], np.array([th]), LEAD, lambda r: r.mu)
    return float(late[0]), float(rem[0])


def main():
    rul = {a.name: a.rul.astype(float) for a in assets.load("pronostia")}
    rows, shift = [], []
    files = sorted(PRED.glob("preds_seed*.npz"))
    assert len(files) == 20, len(files)
    for f in files:
        seed = int(f.stem.replace("preds_seed", ""))
        traj = trajectories(f, rul)
        test = sorted(n for n in traj if not n.startswith("Bearing3"))
        assert len(test) == 10 and not set(traj) & TRAIN, sorted(traj)
        for held in test:
            cal = {n: traj[n] for n in test if n != held}
            th, binding = certify_on(cal)
            age = certified_age([rul[n] for n in cal])
            late, waste = outcome(traj[held], th)
            a_late, a_waste = age_outcome(age, rul[held])
            rows.append({"seed": seed, "asset": held, "theta": float(th), "binding": binding,
                         "late": late, "waste": waste, "age": age, "age_late": a_late, "age_waste": a_waste,
                         "theta_star": lead_value(traj[held])})
        th, binding = certify_on({n: traj[n] for n in test})
        age = certified_age([rul[n] for n in test])
        for n in sorted(set(traj) - set(test)):
            late, waste = outcome(traj[n], th)
            a_late, a_waste = age_outcome(age, rul[n])
            shift.append({"seed": seed, "asset": n, "theta": float(th), "binding": binding, "late": late,
                          "waste": waste, "age_late": a_late, "age_waste": a_waste})

    by = collections.defaultdict(list)
    for r in rows:
        by[r["asset"]].append(r)
    names = sorted(by)
    late_any = sum(any(r["late"] > 0 for r in by[n]) for n in names)
    age_late_any = sum(any(r["age_late"] > 0 for r in by[n]) for n in names)
    res = {
        "n_seeds": len(files), "n_bearings": len(names), "alpha": ALPHA, "lead": LEAD,
        "crc": {"late": float(np.mean([r["late"] for r in rows])),
                "waste": float(np.mean([r["waste"] for r in rows])),
                "bearings_late_any_seed": late_any,
                "cp_upper": float(stats.clopper_pearson(late_any, len(names), alpha=0.05)[1]),
                "per_seed_waste": {s: float(np.mean([r["waste"] for r in rows if r["seed"] == s]))
                                   for s in sorted({r["seed"] for r in rows})}},
        "age": {"late": float(np.mean([r["age_late"] for r in rows])),
                "waste": float(np.mean([r["age_waste"] for r in rows])),
                "bearings_late_any_seed": age_late_any,
                "cp_upper": float(stats.clopper_pearson(age_late_any, len(names), alpha=0.05)[1])},
        "crc_minus_age_waste": paired_summary([np.mean([r["waste"] - r["age_waste"] for r in by[n]])
                                               for n in names]),
        "per_bearing": {n: {"late": float(np.mean([r["late"] for r in by[n]])),
                            "waste": float(np.mean([r["waste"] for r in by[n]])),
                            "age_waste": float(np.mean([r["age_waste"] for r in by[n]])),
                            "theta_star_mean": float(np.mean([r["theta_star"] for r in by[n]]))}
                        for n in names},
        "binding_counts": dict(collections.Counter(r["binding"] for r in rows)),
        "theta_range": [min(r["theta"] for r in rows), max(r["theta"] for r in rows)],
        "verification_floor_10": float(certify.verification_floor(len(names))) if hasattr(certify, "verification_floor") else None,
        "shift_condition3": {"late": float(np.mean([r["late"] for r in shift])),
                             "waste": float(np.mean([r["waste"] for r in shift])),
                             "age_late": float(np.mean([r["age_late"] for r in shift])),
                             "age_waste": float(np.mean([r["age_waste"] for r in shift])),
                             "per_bearing": {n: {k: float(np.mean([r[k] for r in shift if r["asset"] == n]))
                                                 for k in ("late", "waste", "age_late", "age_waste")}
                                             for n in sorted({r["asset"] for r in shift})}},
    }
    (paths.RESULTS / "exact_split_published.json").write_text(json.dumps({"summary": res, "rows": rows,
                                                                         "shift_rows": shift}, indent=1),
                                                              encoding="utf-8")
    c, a, d = res["crc"], res["age"], res["crc_minus_age_waste"]
    print(f"exact split CRC, 10 bearings x 20 seeds: late {c['late']:.3f} (bearings late in any seed "
          f"{late_any}/10, CP upper {c['cp_upper']:.3f}), waste {c['waste']:.3f}")
    print(f"certified age rule, same folds: late {a['late']:.3f} ({age_late_any}/10), waste {a['waste']:.3f}")
    print(f"CRC - age waste per bearing: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
          f"p {d['perm_p_two_sided']:.3f}; thresholds {res['theta_range']}; binding {res['binding_counts']}")
    s = res["shift_condition3"]
    print(f"Condition 3 shift check: late {s['late']:.3f} waste {s['waste']:.3f} | age late {s['age_late']:.3f} "
          f"waste {s['age_waste']:.3f}")
    for n, v in res["per_bearing"].items():
        print(f"  {n}: late {v['late']:.2f} waste {v['waste']:.3f} age {v['age_waste']:.3f} "
              f"theta* {v['theta_star_mean']:.3f}")


if __name__ == "__main__":
    main()
