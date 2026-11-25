"""Certification in service on 124 lithium-ion cells whose capacity is measured at every cycle.

Pre-declared on 2026-11-24 (plan 1c525c8), before any lead-point
value or alarm outcome of this fleet was computed: signal as recorded, no interpolation, lead 30 cycles fixed in advance,
the data authors' own curation, criteria D1-D3.

The second part is post hoc and not blind, prompted by a check of in_service_certification.py: the last recorded value
held between records on the cracks and the ISU-ILCC cells, bounds with the duty as the unit, and the law of the age
count given its ties.

    python experiments/in_service_matr.py        # writes results/in_service_matr.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import paths, service as S, stats  # noqa: E402
from certalarm.data import fleets  # noqa: E402
from experiments import in_service_certification as X  # noqa: E402

LEAD_CYCLES = 30.0


def matr_rows(noisy=False, every=1):
    out, policy = [], {}
    for cell, (cyc, q, life, pol, batch) in fleets.matr(noisy).items():
        if noisy and not np.isfinite(life):       # a noisy channel without a cycle-life label (b3c23)
            continue
        assert np.isfinite(life), cell
        keep = np.isfinite(q) & (q > 0.1) & (cyc <= life)
        t, c = cyc[keep][::every], q[keep][::every]
        out.append((cell, batch, t, {"condition": c, "age": -t}, life))
        policy[cell] = pol
    return out, policy


def clustered(fleet, in_cal, in_test, unit):
    """Late test assets and late test units (a unit is late if any of its assets is), with the one-sided 97.5% bounds."""
    th = S.threshold([a.lead_value for a in fleet if in_cal(a)], 0)
    late, units = [], {}
    for a in fleet:
        if in_test(a):
            x = bool(a.lead_value >= th)
            late.append(x)
            units[unit(a)] = units.get(unit(a), False) or x
    cp = lambda v: stats.clopper_pearson(int(sum(v)), len(v))[1]
    return {"late": int(sum(late)), "m": len(late), "bound": cp(late), "late_units": int(sum(units.values())),
            "units": len(units), "bound_units": cp(list(units.values()))}


def held(rows, records):
    """The same inspections with the last recorded value held until the next record."""
    out = []
    for n, g, t, s, T in rows:
        rt, rv = records(n)
        i = np.searchsorted(rt, t, side="right") - 1
        out.append((n, g, t, {"condition": rv[i], "age": -t}, T))
    return out


def tie_law(values, n_perm=20_000, seed=0):
    """Number of late assets at rank 0 over random orders of one set of values: the law given its ties."""
    rng, v = np.random.default_rng(seed), np.asarray(values)
    counts = np.empty(n_perm, int)
    for i in range(n_perm):
        x = v[rng.permutation(len(v))]
        m = np.maximum.accumulate(x)
        counts[i] = 1 + int((x[1:] > m[:-1]).sum())
    return counts


def pair(rows, lead="group", only=None):
    rows = [r for r in rows if only is None or only(r)]
    tabs = {s: X.Table(X.build(rows, s, lead=lead)) for s in ("condition", "age")}
    w = {s: t.loo() for s, t in tabs.items()}
    runs = {s: S.run_in_service(X.build(rows, s, lead=lead), 0) for s in tabs}
    return {"assets": len(rows), **{s: float(w[s].mean()) for s in w}, "pair": X.summarise_pairs(w["condition"] - w["age"]),
            "late": {s: X.against_law(sum(q["late"] for q in runs[s]), S.record_probs(len(rows), 0)) for s in runs}}


def main():
    rows, policy = matr_rows()
    assert len(rows) == 124, len(rows)
    b = lambda *g: (lambda a: a.group in g)
    transfer = {"batch 1 -> 2": (b(1), b(2)), "batches 1-2 -> 3": (b(1, 2), b(3))}
    res = {"matr": X.analyse(rows, ("condition", "age"), transfer, lead=LEAD_CYCLES)}
    r = res["matr"]
    r["policies"] = len(set(policy.values()))
    r["batches"] = {k: sum(x[1] == k for x in rows) for k in (1, 2, 3)}
    for s in ("condition", "age"):
        fl = X.build(rows, s, lead=LEAD_CYCLES)
        for nm, (f, g) in transfer.items():
            r["o5"][s][nm]["clustered"] = clustered(fl, f, g, lambda a: policy[a.name])
        # the same batch-1 threshold kept for every later cell (the series behind the figure)
        th = S.threshold([a.lead_value for a in fl if a.group == 1], 0)
        r["signals"][s]["frozen_batch1_late_at"] = [i + 1 for i, a in enumerate(fl) if a.group > 1 and a.lead_value >= th]
    r["o2_by_policy"] = {}
    tabs = {s: X.Table(X.build(rows, s, lead=LEAD_CYCLES)).loo() for s in ("condition", "age")}
    d, pol = tabs["condition"] - tabs["age"], np.array([policy[x[0]] for x in rows])
    r["o2_by_policy"] = X.summarise_pairs([d[pol == p].mean() for p in sorted(set(pol))])
    inside = lambda o: o["range95"][0] <= o["late"] <= o["range95"][1]
    o2 = r["o2"]["condition_minus_age"]
    res["criteria"] = {"D1": o2["mean"] < 0 and o2["p"] < 0.05, "D2": inside(r["signals"]["condition"]["o1"][0]),
                       "D3": inside(r["signals"]["age"]["o1"][0])}
    res["sensitivity"] = {"lead 15": pair(rows, 15.0), "lead 60": pair(rows, 60.0),
                          "every 10th cycle": pair(matr_rows(every=10)[0], LEAD_CYCLES), "every 50th cycle": pair(matr_rows(every=50)[0], LEAD_CYCLES),
                          "noisy channels kept": pair(matr_rows(noisy=True)[0], LEAD_CYCLES)}

    # ---- post hoc
    a, N = fleets.virkler()
    cells = fleets.isu_ilcc()
    post = {"cracks, last record held": pair(held(X.virkler(), lambda n: (N[:, int(n[1:]) - 1], -a))),
            "cells, last record held": pair(held(X.isu(), lambda n: (cells[n][0], cells[n][1] / cells[n][1][0])))}
    post["cells, last record held, without the two duties failing before their first test"] = pair(
        held(X.isu(), lambda n: (cells[n][0], cells[n][1] / cells[n][1][0])), only=lambda q: q[1] not in (11, 14))
    iso = X.build(X.isu(), "condition")
    isa = X.build(X.isu(), "age")
    post["cells, duty as the unit"] = {"condition": clustered(iso, lambda q: q.group <= 29, lambda q: q.group >= 30, lambda q: q.group),
                                       "age": clustered(isa, lambda q: q.group <= 29, lambda q: q.group >= 30, lambda q: q.group),
                                       "fleet_bound_duties": 1 - 0.025 ** (1 / len({q.group for q in iso}))}
    w = X.Table(iso).loo() - X.Table(isa).loo()
    g = np.array([q.group for q in iso])
    post["cells, C2 over duty means"] = X.summarise_pairs([w[g == k].mean() for k in sorted(set(g))])
    post["regime_bound_cells"] = float(sum(S.record_probs(int((g == k).sum())).sum() for k in set(g)))
    post["tie_law_age"] = {}
    for nm, rw in (("virkler", X.virkler()), ("isu", X.isu()), ("cmapss", X.turbofans())):
        v = [q.lead_value for q in X.build(rw, "age")]
        c = tie_law(v)
        obs = sum(q["late"] for q in S.run_in_service(X.build(rw, "age"), 0))
        post["tie_law_age"][nm] = {"late": int(obs), "mean": float(c.mean()), "range95": [int(np.percentile(c, 2.5)), int(np.percentile(c, 97.5))],
                                   "p_upper": float((c >= obs).mean()), "p_two_sided": float(min(1.0, 2 * min((c >= obs).mean(), (c <= obs).mean())))}
    ov = {}
    for s in ("condition", "age"):
        fl = X.build(rows, s, lead=LEAD_CYCLES)
        ov[s] = {k: {"late": int(sum(q["late"] for q in x)), "without_alarm": int(sum(q["theta"] == -np.inf for q in x)),
                     "bound": float(S.record_probs(len(fl), k - 1).sum())} for k in (1, 4, 16) for x in [S.run_overlapping(fl, k)]}
    post["matr_overlap"] = ov
    fl = X.build(rows, "condition", lead=LEAD_CYCLES)
    post["batch_lives"] = {k: {"median_life": float(np.median([q.T for q in fl if q.group == k])),
                               "median_lead_capacity": float(np.median([q.lead_value for q in fl if q.group == k]))} for k in (1, 2, 3)}
    res["post_hoc"] = post
    (paths.RESULTS / "in_service_matr.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    print(f"MATR: {r['assets']} cells, batches {r['batches']}, {r['policies']} policies, life {r['life_range']}, median {r['median_life']}")
    for s, o in r["signals"].items():
        q = o["o1"][0]
        print(f"  {s:<9} rank 0 late {q['late']} (expected {q['expected']:.2f}, {q['range95']}, p {q['p_two_sided']:.3f}) at {q['late_at']}; LOO waste {o['loo_waste']:.3f} ties {o['ties']} inf {o['infinite']}")
        print("     ranks", {k: (x['late'], round(x['expected'], 1), x['range95']) for k, x in o["o1"].items()})
        print("     waste vs n", {n: round(x["rank0"], 3) for n, x in o["o3"].items()}, "level", {n: round(x["level"], 3) for n, x in o["o3"].items()})
        print("     cost r=10", {k: round((x["cost"][10][0] / x["cost"][10][1]) / o["run_to_failure_rate"][10], 3) for k, x in o["o1"].items()},
              "random", {k: round(x[10], 3) for k, x in o["random_order"].items()},
              "fixed", {n: (x["failures"], round((x["cost"][10][0] / x["cost"][10][1]) / o["run_to_failure_rate"][10], 3)) for n, x in o["fixed"].items()}, "O6", o["o6"])
    print("  O2", r["o2"], "\n  by policy", r["o2_by_policy"])
    for s, t in r["o5"].items():
        for nm, x in t.items():
            print("  O5", s, nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k != "clustered"}, x["clustered"])
    print("criteria", res["criteria"])
    for nm, x in res["sensitivity"].items():
        print("  sens", nm, x["assets"], round(x["condition"], 3), round(x["age"], 3), round(x["pair"]["mean"], 3), x["pair"]["p"], x["late"]["condition"]["late"], x["late"]["age"]["late"])
    for nm, x in post.items():
        print("  post", nm, json.dumps(x, default=float)[:420])


if __name__ == "__main__":
    main()
