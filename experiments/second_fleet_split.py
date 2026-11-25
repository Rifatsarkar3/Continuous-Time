"""Exact split certification on XJTU-SY, and a certified RMS baseline on both fleets.

Pre-declared on 2026-11-24 (plan 36a250c), before
any outcome was computed. Descriptive.

Every rule alarms at the first minute whose signal falls strictly below a threshold, so all rules share the
lead-point reduction: theta*_i is the signal's running minimum up to the last minute with remaining life >= lead.
The certified threshold is the exact order statistic, just above the (k_n + 1)-th largest calibration value
(np.nextafter), no grid. Rules: network (PRONOSTIA: Monte Carlo mean, as stored; XJTU-SY: deterministic output of the
published final models), the two pre-declared RMS signals, and the certified age rule on the same folds.

S2 needs the mean RMS of the first 10 minutes, so it is undefined before then and cannot fire there.

    python experiments/second_fleet_split.py                  # registered: XJTU-SY `final` (5 seeds) -> results/second_fleet_split.json
    python experiments/second_fleet_split.py --xjtu final20   # post hoc: the published, public 20-seed set -> ..._final20.json

The pre-declaration named the `final` files; the published XJTU-SY results and the public repository hold `final20`
(found after the registered run). Both are reported.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, paths  # noqa: E402
from certalarm.predictors.dsaf_train import to_minutes  # noqa: E402
from experiments.age_baseline import age_outcome, certified_age, last_on_time  # noqa: E402
from experiments.c1_leadpoint_xjtu import paired_summary  # noqa: E402
from experiments.dev_hybrid_screen import rms_per_minute  # noqa: E402

ALPHA, LEAD = 0.10, 0.05
DS = paths.DUAL_STREAM / "outputs"
PRON_PRED = DS / "r1_final_B10" / "final20"
XJTU_VARIANT = sys.argv[sys.argv.index("--xjtu") + 1] if "--xjtu" in sys.argv else "final"
XJTU_PRED = DS / "exp17_xjtu" / XJTU_VARIANT
BASE_MIN = 10                                     # minutes of the S2 baseline


def name_of(key):
    return "Bearing" + key.split("Bearing")[1]


def certified_threshold(cal):
    """Just above the (k_n + 1)-th largest calibration lead-point value; None below the certifiability floor."""
    k = int(np.floor(ALPHA * (len(cal) + 1) - 1))
    if k < 0:
        return None
    return np.nextafter(sorted(cal, reverse=True)[k], np.inf)


def fire(sig, theta):
    hit = np.flatnonzero(sig < theta)
    return int(hit[0]) if hit.size else None


def outcome(sig, rul, theta):
    i = fire(sig, theta)
    if i is None:
        return 1.0, 0.0
    return float(rul[i] < LEAD), float(rul[i])


def lead_value(sig, rul):
    return float(np.min(sig[:last_on_time(rul) + 1]))


def rms_signals(asset):
    r = np.asarray(rms_per_minute(asset), float)
    s1 = -r
    base = float(r[:BASE_MIN].mean())
    s2 = np.full(len(r), np.inf)                  # undefined, so never below a threshold, before BASE_MIN
    s2[BASE_MIN:] = -np.log(r[BASE_MIN:] / base)
    return {"S1": s1, "S2": s2}


def network_signals(fleet, rul):
    """-> {seed: {bearing: minute signal}} of the published predictors (PRONOSTIA: MC mean; XJTU-SY: raw)."""
    out = {}
    if fleet == "pronostia":
        for f in sorted(PRON_PRED.glob("preds_seed*.npz")):
            d = np.load(f)
            out[int(f.stem.replace("preds_seed", ""))] = {
                name_of(k): to_minutes(d[k], d[k.replace("mcmean_", "mcstd_")], assets.PER_MIN["pronostia"])[0]
                for k in d.files if k.startswith("mcmean_test_")}
    else:
        for f in sorted(XJTU_PRED.glob("preds_seed*.npz")):
            seed = int(f.stem.replace("preds_seed", ""))
            zf = XJTU_PRED / f"zeroshot_preds_seed{seed}.npz"
            d, z = np.load(f), np.load(zf)
            pairs = {name_of(k): d[k] for k in d.files if k.startswith("raw_test_")}
            pairs.update({name_of(k): z[k] for k in z.files if k.startswith("raw_zeroshot_")})
            out[seed] = {n: to_minutes(v, np.zeros_like(v), assets.PER_MIN["xjtu"])[0] for n, v in pairs.items()}
    for s, d in out.items():
        for n, v in d.items():
            assert len(v) == len(rul[n]), (fleet, s, n, len(v), len(rul[n]))
    return out


def loo(names, sig, rul, age_rul):
    """Leave-one-out over names: exact threshold from the others; -> per bearing (late, waste, theta, binding)."""
    stars = {n: lead_value(sig[n], rul[n]) for n in names}
    out = {}
    for h in names:
        cal = {n: stars[n] for n in names if n != h}
        th = certified_threshold(list(cal.values()))
        late, waste = outcome(sig[h], rul[h], th)
        binding = max(cal, key=cal.get)
        a = certified_age([age_rul[n] for n in names if n != h])
        al, aw = age_outcome(a, rul[h])
        out[h] = {"late": late, "waste": waste, "theta": float(th), "binding": binding,
                  "age_late": al, "age_waste": aw, "age": a}
    return out


def main():
    res, rows = {}, []
    for fleet in ("pronostia", "xjtu"):
        a = {x.name: x for x in assets.load(fleet)}
        rul = {n: np.asarray(x.rul, float) for n, x in a.items()}
        nets = network_signals(fleet, rul)
        names = sorted(next(iter(nets.values())))
        assert len(names) == {"pronostia": 10, "xjtu": 11}[fleet], (fleet, len(names))
        rms = {n: rms_signals(a[n]) for n in names}
        signals = {"network": {s: {n: nets[s][n] for n in names} for s in nets}}
        signals.update({k: {0: {n: rms[n][k] for n in names}} for k in ("S1", "S2")})
        per, seeds_of = {}, {}
        for rule, by_seed in signals.items():
            for seed, sig in by_seed.items():
                r = loo(names, sig, rul, rul)
                for n, v in r.items():
                    rows.append({"fleet": fleet, "rule": rule, "seed": seed, "asset": n, **v})
        age_bearing = {n: None for n in names}
        # age rows repeat in every loo call, so take them once from the first rule / seed
        for r0 in (x for x in rows if x["fleet"] == fleet and x["rule"] == "network"):
            age_bearing[r0["asset"]] = r0["age_waste"]
        summary = {"n": len(names), "condition3": sum(n.startswith("Bearing3") for n in names)}
        means = {}
        for rule in signals:
            rr = [x for x in rows if x["fleet"] == fleet and x["rule"] == rule]
            seeds = sorted({x["seed"] for x in rr})
            late_per_seed = [sum(x["late"] for x in rr if x["seed"] == s) for s in seeds]
            waste_b = collections.defaultdict(list)
            for x in rr:
                waste_b[x["asset"]].append(x["waste"])
            means[rule] = {n: float(np.mean(v)) for n, v in waste_b.items()}
            summary[rule] = {"late": float(np.mean([x["late"] for x in rr])), "waste": float(np.mean([x["waste"] for x in rr])),
                             "late_bearings_per_seed": [float(x) for x in late_per_seed],
                             "binding": dict(collections.Counter(x["binding"] for x in rr))}
        summary["age"] = {"late": float(np.mean([x["age_late"] for x in rows if x["fleet"] == fleet and x["rule"] == "network"])),
                          "waste": float(np.mean(list(age_bearing.values())))}
        d = lambda p, q: paired_summary([means[p][n] - (means[q][n] if q != "age" else age_bearing[n]) for n in names])
        summary["paired"] = {"network_minus_age": d("network", "age"), "network_minus_S1": d("network", "S1"),
                             "network_minus_S2": d("network", "S2"),
                             "S1_minus_age": d("S1", "age"), "S2_minus_age": d("S2", "age")}
        summary["per_bearing_waste"] = {n: {"network": means["network"][n], "S1": means["S1"][n], "S2": means["S2"][n],
                                            "age": age_bearing[n]} for n in names}
        res[fleet] = summary
        print(f"\n{fleet}: {len(names)} bearings ({summary['condition3']} in Condition 3)")
        for rule in ("network", "S1", "S2", "age"):
            s = summary[rule]
            print(f"  {rule:<8} late {s['late']:.3f} waste {s['waste']:.3f}" + (
                f"  late per seed {s['late_bearings_per_seed'][:3]}... binding {s['binding']}" if "binding" in s else ""))
        for k, v in summary["paired"].items():
            print(f"  {k:<20} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] p {v['perm_p_two_sided']:.3f} (lower on {v['lp_lower']} of {v['n']})")
    # E1: at k_n = 0 exactly one late bearing per seed and rule, unless two lead-point values tie
    name = "second_fleet_split" + ("" if XJTU_VARIANT == "final" else f"_{XJTU_VARIANT}")
    (paths.RESULTS / f"{name}.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1, default=float),
                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
