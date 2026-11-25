"""Certified age replacement on the bearing fleets, the baseline every condition-based rule must beat.

Descriptive (requested in review, 2026-11-19); the bearing test sets were read before. Written and
committed before it was run.

An age rule alarms at a fixed age a, the same for every asset. It needs no trained predictor, so
split conformal risk control applies exactly with every non-test bearing as a calibration asset.
By the lead-point reduction, with t*_i the last minute with remaining life >= lead, the
certified age is the (k_n + 1)-th smallest t*_i over the calibration bearings: the alarm is late
for a test bearing iff a > t*_test. The same outer folds and seeds as the stored arms are used, so
every comparison is paired by (seed, fold, bearing). Observation is full (the age rule reads no
measurement).

Reported per fleet: pooled realized risk and wasted life over all test rows; paired per-bearing
differences of certified wasted life against each stored conformal arm (full observation rows),
with 95% t-intervals and exact sign-flip p; and the long-run cost rate of each rule under renewal
reward: a cycle ends at the alarm (preventive cost 1) if the alarm is on time, and at failure
(cost c) if it is late, so the rate is sum(cost) / sum(cycle length in minutes) over test rows,
reported at c = 5, 10, 20 with asset-clustered bootstrap intervals for each rate and for the
paired difference age minus conformal.

Revised 2026-11-19 after an internal audit: a conformal row's late value is a fraction when its
outcome averages inner models, and it had been counted as a full failure. Cost and cycle length are
now the expectations over the averaged outcomes (see cost_terms); the age rule's rows are binary
and unchanged.

    python experiments/age_baseline.py        # writes results/age_baseline.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, checkpoint, crossfit, evaluate, paths  # noqa: E402
from experiments.c1_leadpoint_xjtu import paired_summary  # noqa: E402

ALPHA, LEAD, COSTS = 0.10, 0.05, (5, 10, 20)
FLEETS = {"pronostia": ((42, 123, 777, 999, 7), {"CT-GRU": "pron_ctgru_pronostia_ctgru",
                                                  "DSAFLite": "pron_dsaflite_pronostia_dsaflite",
                                                  "DANN": "pron_dann_pronostia_dsaflite_dann"}),
          "xjtu": ((42, 123, 777), {"CT-GRU": "xjtu_ctgru_xjtu_ctgru",
                                    "DSAFLite": "xjtu_dsaflite_xjtu_dsaflite"})}


def last_on_time(rul):
    ok = np.flatnonzero(np.asarray(rul, float) >= LEAD)
    return int(ok[-1]) if ok.size else -1


def age_outcome(a, rul):
    """Alarm at minute a: late iff a > t*; wasted life is the true remaining life at minute a."""
    late = a > last_on_time(rul)
    waste = float(rul[a]) if a < len(rul) else 0.0
    return float(late), waste


def certified_age(cal_rul):
    t = np.sort([last_on_time(r) for r in cal_rul])
    k = int(np.floor(ALPHA * (len(t) + 1) - 1))
    return int(t[k]) if k >= 0 else None          # the (k+1)-th smallest lead-point age


def cost_terms(r, c):
    """Expected cost and cycle length of one test row whose outcome may average several models.

    With late fraction L and mean remaining life at the alarm W, the cost is c L + (1 - L). An on-time
    alarm ends the cycle at the alarm, a late one at failure (length T), so the expected length is
    T (1 - W + L w_late), w_late being the mean remaining life at the late alarms. For L = 1 that gives T
    exactly; for 0 < L < 1, w_late (below the lead, 0.05) is not stored and is neglected, which
    understates the length by less than 0.05 L T.
    """
    L, W, T = float(r["late"]), float(r["waste"]), float(r["T"])
    return c * L + (1.0 - L), T if L >= 1.0 else T * (1.0 - W)


def cost_rate(rows, c):
    terms = [cost_terms(r, c) for r in rows]
    return sum(t[0] for t in terms) / sum(t[1] for t in terms)


def boot_rates(rules, c, n_boot=2000, seed=0):
    """Asset-clustered bootstrap: resample bearings, recompute every rule's rate and age minus each rule."""
    rng = np.random.default_rng(seed)
    by = {k: collections.defaultdict(list) for k in rules}
    for k, rows in rules.items():
        for r in rows:
            by[k][r["asset"]].append(cost_terms(r, c))
    names = sorted(set.intersection(*(set(v) for v in by.values())))
    draws = {k: [] for k in rules}
    for _ in range(n_boot):
        pick = rng.choice(len(names), len(names), replace=True)
        for k in rules:
            t = [x for i in pick for x in by[k][names[i]]]
            draws[k].append(sum(a for a, _ in t) / sum(b for _, b in t))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {k: ci(v) for k, v in draws.items()}
    out.update({f"age_minus_{k}": ci(np.subtract(draws["age"], v)) for k, v in draws.items() if k != "age"})
    return out


def main():
    out = {}
    for fleet, (seeds, stored) in FLEETS.items():
        by = {a.name: a for a in assets.load(fleet)}
        rul = {n: np.asarray(a.rul, float) for n, a in by.items()}
        rows = []
        for seed in seeds:
            for i, spec in enumerate(crossfit.build_fold(list(by.values()), seed=seed)):
                a = certified_age([rul[n] for n in spec["calibration"]])
                for t in spec["test"]:
                    late, waste = age_outcome(a, rul[t])
                    rows.append({"seed": seed, "fold": i, "asset": t, "age": a, "late": late,
                                 "waste": waste, "T": len(rul[t])})
        keyed = {f"{r['seed']}/{r['fold']}/{r['asset']}": r for r in rows}
        s = evaluate.summarize(keyed, cluster_key={k: r["asset"] for k, r in keyed.items()})
        res = {"late": float(np.mean([r["late"] for r in rows])),
               "waste": float(np.mean([r["waste"] for r in rows])), "rows": len(rows),
               "boot_upper": s["late_upper_bootstrap"], "cp_upper": s["late_upper_binomial"],
               "cost_rate": {}, "paired_vs": {}}
        age_b = collections.defaultdict(list)
        for r in rows:
            age_b[r["asset"]].append(r["waste"])
        T = {n: len(r) for n, r in rul.items()}
        rules = {"age": rows}
        for arm, tag in stored.items():
            st = [r for rs in checkpoint.completed(tag).values() for r in rs
                  if r["rule"] == "crc" and r["process"] == "full" and r["seed"] in seeds]
            rules[f"crc_{arm}"] = [{"asset": r["asset"], "late": r["late"], "waste": r["waste"],
                                    "T": T[r["asset"]]} for r in st]
            res.setdefault("fractional_late_rows", {})[arm] = sum(0 < r["late"] < 1 for r in st)
            crc_b = collections.defaultdict(list)
            for r in st:
                crc_b[r["asset"]].append(r["waste"])
            names = sorted(set(age_b) & set(crc_b))
            res["paired_vs"][arm] = {
                "crc_late": float(np.mean([r["late"] for r in st])),
                "crc_waste": float(np.mean([r["waste"] for r in st])),
                "age_minus_crc_waste": paired_summary([np.mean(age_b[n]) - np.mean(crc_b[n]) for n in names])}
        res["cost_rate_ci95"] = {}
        for c in COSTS:
            res["cost_rate"][str(c)] = {k: cost_rate(v, c) * 1000 for k, v in rules.items()}   # per 1000 min
            res["cost_rate_ci95"][str(c)] = {k: [x * 1000 for x in v] for k, v in boot_rates(rules, c).items()}
        out[fleet] = res
        print(f"\n{fleet}: age rule late {res['late']:.3f} waste {res['waste']:.3f} ({len(rows)} rows)")
        for arm, p in res["paired_vs"].items():
            d = p["age_minus_crc_waste"]
            print(f"  vs CRC {arm:<8} (late {p['crc_late']:.3f} waste {p['crc_waste']:.3f}): age - crc waste "
                  f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] p {d['perm_p_two_sided']:.3f}")
        for c, v in res["cost_rate"].items():
            ci = res["cost_rate_ci95"][c]
            print(f"  cost rate per 1000 min at c = {c}: " + ", ".join(
                f"{k} {x:.1f} [{ci[k][0]:.1f}, {ci[k][1]:.1f}]" for k, x in v.items()))
            print("    age minus: " + ", ".join(f"{k[10:]} [{a:+.1f}, {b:+.1f}]" for k, (a, b) in ci.items()
                                              if k.startswith("age_minus")))
    (paths.RESULTS / "age_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwritten to results/age_baseline.json")


if __name__ == "__main__":
    main()
