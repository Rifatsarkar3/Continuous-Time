"""Certification in service on five fleets, two of them new (68 fatigue cracks, 251 battery cells).

Pre-declared on 2026-11-24 (plan bf4e122), before any
lead-point value or alarm outcome of the new fleets was computed. Virkler and ISU-ILCC are blind; PRONOSTIA, XJTU-SY and
C-MAPSS had been analysed before and are descriptive. Outcomes O1-O6 and criteria C1-C4 follow that document.

    python experiments/in_service_certification.py        # writes results/in_service.json
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, cmapss, paths, service as S  # noqa: E402
from certalarm.data import fleets  # noqa: E402
from certalarm.stats import holm  # noqa: E402
from experiments.c2_ridge_cmapss import make_ridge  # noqa: E402
from experiments.e4_asset_count import partition  # noqa: E402
from experiments.second_fleet_split import rms_signals  # noqa: E402

ALPHA, LEAD, RANKS, COSTS = 0.10, 0.05, (0, 1, 2, 4), (10, 5, 20)
N_PERM, N_FLIP = 2000, 100_000


# --- fleets: name -> list of (asset name, group, inspection times, {signal: values}, life) -----------------------------

def virkler():
    a, N = fleets.virkler()
    out = []
    for j in range(N.shape[1]):
        T = float(N[-1, j])
        t = np.arange(0.0, T + 1e-9, 1000.0)
        out.append((f"V{j + 1}", 1, t, {"condition": -np.interp(t, N[:, j], a), "age": -t}, T))
    return out


def isu(eol=0.80, daily=True):
    out = []
    for cell, (days, cap) in fleets.isu_ilcc().items():
        rel = cap / cap[0]
        hit = np.flatnonzero(rel < eol)
        if not hit.size:
            continue
        j = int(hit[0])
        if daily:
            T = float(np.interp(eol, [rel[j], rel[j - 1]], [days[j], days[j - 1]]))
            t = np.arange(0.0, T + 1e-9, 1.0)
            c = np.interp(t, days, rel)
        else:
            T, t, c = float(days[j]), days[: j + 1], rel[: j + 1]
        out.append((cell, fleets.isu_key(cell)[0], t, {"condition": c, "age": -t}, T))
    return out


def bearings(fleet):
    out = []
    for a in assets.load(fleet):
        t = np.arange(float(a.n_min))
        out.append((a.name, a.condition, t, {**rms_signals(a), "age": -t}, float(a.n_min - 1)))
    return out


def turbofans():
    train, pool, test, _ = partition(cmapss.load(), 42)
    predict = make_ridge(train)[0]
    out = []
    for u in sorted(pool + test, key=lambda u: (u.condition, int(u.name.split("_u")[1]))):
        t = np.arange(float(u.n_min))
        out.append((u.name, u.condition, t, {"condition": np.asarray(predict(u), float), "age": -t}, float(u.n_min - 1)))
    return out


def build(rows, signal, lead="group", factor=1.0):
    """-> [Asset]; lead = factor x 0.05 x median life of the operating condition, or of the asset's own life, or a number."""
    med = {g: np.median([r[4] for r in rows if r[1] == g]) for g in {r[1] for r in rows}}
    tau = lambda g, T: float(lead) if isinstance(lead, (int, float)) else factor * LEAD * (med[g] if lead == "group" else T)
    return [S.Asset(n, g, t, s[signal], T, tau(g, T)) for n, g, t, s, T in rows]


# --- fast crossing tables for the many-threshold computations ----------------------------------------------------------

class Table:
    """Alarm outcomes of every asset at given thresholds, by default just above every lead-point value of the fleet."""

    def __init__(self, fleet, th=None):
        self.N = len(fleet)
        self.values = np.array([a.lead_value for a in fleet])
        self.T = np.array([a.T for a in fleet])
        self.order = np.argsort(-self.values, kind="stable")          # decreasing
        th = np.nextafter(self.values, np.inf) if th is None else np.asarray(th, float)
        self.waste, self.time, self.late = (np.zeros((self.N, len(th))) for _ in range(3))
        for i, a in enumerate(fleet):
            neg = -np.minimum.accumulate(a.s)
            idx = np.searchsorted(neg, -th, side="right")             # first inspection below the threshold
            hit = idx < len(a.s)
            at = np.where(hit, a.t[np.minimum(idx, len(a.s) - 1)], np.nan)
            late = ~hit | (a.T - np.nan_to_num(at) < a.tau)
            self.waste[i] = np.where(hit, (a.T - np.nan_to_num(at)) / a.T, 0.0)
            self.late[i] = late
            self.time[i] = np.where(late, a.T, np.nan_to_num(at) + a.tau)

    def expected(self, n, k, what):
        """Mean over assets of the exact expectation of `what` when n of the other assets calibrate rank k."""
        w = S.order_weights(self.N - 1, n, k)
        M = getattr(self, what)
        return float(np.mean([M[i, self.order[self.order != i]] @ w for i in range(self.N)]))

    def loo(self, k=0):
        """Wasted life of each asset under the rank-k threshold of all the others."""
        return np.array([self.waste[i, self.order[self.order != i][k]] for i in range(self.N)])


def summarise_pairs(d):
    d = np.asarray(d, float)
    n, se = len(d), d.std(ddof=1) / np.sqrt(len(d))
    t = st.t.ppf(0.975, n - 1)
    if n <= 20:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        p = float(np.mean(np.abs(signs @ d / n) >= abs(d.mean()) - 1e-12))
    else:
        rng = np.random.default_rng(0)
        hits = sum(int((np.abs(rng.choice((-1.0, 1.0), (10_000, n)) @ d / n) >= abs(d.mean()) - 1e-12).sum())
                   for _ in range(N_FLIP // 10_000))
        p = (hits + 1) / (N_FLIP + 1)
    return {"n": n, "mean": float(d.mean()), "ci95": [float(d.mean() - t * se), float(d.mean() + t * se)], "p": p,
            "lower": int((d < -1e-12).sum()), "higher": int((d > 1e-12).sum())}


def against_law(x, p):
    pmf = S.count_pmf(p)
    cdf = np.cumsum(pmf)
    lo, hi = float(cdf[x]), float(1.0 - (cdf[x - 1] if x else 0.0))
    return {"late": int(x), "expected": float(np.sum(p)), "range95": [int(np.searchsorted(cdf, 0.025)),
            int(np.searchsorted(cdf, 0.975))], "p_two_sided": min(1.0, 2 * min(lo, hi)), "p_upper": hi}


def campaign(rows, r):
    """Cost and operating time of a list of outcomes, a failure costing r planned replacements."""
    return sum(r if x["late"] else 1 for x in rows), sum(x["time"] for x in rows)


def fixed_after(fleet, n0):
    """Run n0 assets to failure, then keep the rank-0 threshold of those n0 for the rest."""
    th = S.threshold([a.lead_value for a in fleet[:n0]], 0)
    out = [{"late": True, "waste": 0.0, "time": a.T} for a in fleet[:n0]]
    for a in fleet[n0:]:
        i = a.crossing(th)
        late = i is None or a.T - a.t[i] < a.tau
        out.append({"late": late, **S._outcome(a, i, late)})
    return out


def analyse(rows, signals, transfer, **kw):
    res = {"assets": len(rows), "median_life": float(np.median([r[4] for r in rows])),
           "life_range": [float(min(r[4] for r in rows)), float(max(r[4] for r in rows))], "signals": {}}
    ns = [n for n in (9, 12, 15, 18, 36, 72, 144, 288) if n < len(rows) - 1] + [len(rows) - 1]
    tables = {}
    for sig in signals:
        fleet = build(rows, sig, **kw)
        tab = tables[sig] = Table(fleet)
        out = {"ties": len(tab.values) - len(np.unique(tab.values)), "infinite": int(np.isinf(tab.values).sum())}
        out["o1"] = {}
        for k in RANKS:
            run = S.run_in_service(fleet, k)
            for t, x in enumerate(run):                               # S1 on real signals
                assert x["theta"] == (S.threshold(tab.values[:t], k) if t > k else -np.inf)
            x = sum(q["late"] for q in run)
            out["o1"][k] = {**against_law(x, S.record_probs(len(fleet), k)),
                            "late_at": [t + 1 for t, q in enumerate(run) if q["late"]],
                            "mean_waste_after_start": float(np.mean([q["waste"] for q in run[k + 1:]])),
                            "cost": {r: [float(c) for c in campaign(run, r)] for r in COSTS}}
        w = tab.loo()
        out["loo_waste"], out["loo_late"] = float(w.mean()), int(sum(
            tab.late[i, tab.order[tab.order != i][0]] for i in range(tab.N)))
        out["o3"] = {n: {**{f"rank{k}": tab.expected(n, k, "waste") for k in RANKS if k < n},
                         "level": tab.expected(n, max(0, S.level_rank(n, ALPHA)), "waste")} for n in ns}
        fixed = {n0: fixed_after(fleet, n0) for n0 in (9, 36) if n0 < len(fleet)}
        out["fixed"] = {n0: {"failures": int(sum(q["late"] for q in f)), "cost": {r: [float(c) for c in campaign(f, r)]
                        for r in COSTS}} for n0, f in fixed.items()}
        lvl = S.run_level_in_service(fleet, ALPHA)
        full = []
        for t, a in enumerate(fleet):
            th = S.threshold(tab.values[:t], max(0, S.level_rank(t, ALPHA))) if t else -np.inf
            i = a.crossing(th)
            late = i is None or a.T - a.t[i] < a.tau
            full.append({"late": late, **S._outcome(a, i, late)})
        out["o6"] = {nm: {"late": int(sum(q["late"] for q in x)), "mean_waste": float(np.mean([q["waste"] for q in x[1:]]))}
                     for nm, x in (("level_bounds", lvl), ("level_complete_records", full))}
        out["o6"]["rank0"] = {"late": out["o1"][0]["late"], "mean_waste": out["o1"][0]["mean_waste_after_start"]}
        rng = np.random.default_rng(0)
        acc = {k: [] for k in RANKS}
        for _ in range(N_PERM):
            perm = rng.permutation(tab.N)
            for k in RANKS:
                acc[k].append(permuted_cost(tab, perm, k))
        out["random_order"] = {k: {r: float(np.mean([((r - 1) * f + tab.N) / tm for f, tm in acc[k]])
                                            / (tab.N * r / tab.T.sum())) for r in COSTS}
                               | {"failures": float(np.mean([f for f, _ in acc[k]]))} for k in RANKS}
        out["run_to_failure_rate"] = {r: float(tab.N * r / tab.T.sum()) for r in COSTS}
        res["signals"][sig] = out
    res["o2"] = {f"{s}_minus_age": summarise_pairs(tables[s].loo() - tables["age"].loo()) for s in signals if s != "age"}
    res["o5"] = {s: {nm: transfer_test(build(rows, s, **kw), f, g) for nm, (f, g) in transfer.items()} for s in signals}
    return res


def permuted_cost(tab, perm, k):
    """(failures, operating time) of the rank-k rule in service on one random order, from the crossing tables."""
    top, fails, time = [], 0, 0.0                                     # min-heap of the k+1 largest (value, index)
    for i in perm:
        if len(top) <= k:
            late, tm = True, tab.T[i]
        else:
            j = top[0][1]
            late, tm = bool(tab.late[i, j]), tab.time[i, j]
        fails += late
        time += tm
        if late:
            heapq.heappush(top, (tab.values[i], int(i)))
            if len(top) > k + 1:
                heapq.heappop(top)
    return fails, time


def transfer_test(fleet, in_cal, in_test):
    cal, test = [a for a in fleet if in_cal(a)], [a for a in fleet if in_test(a)]
    th = S.threshold([a.lead_value for a in cal], 0)
    x = int(sum(a.lead_value >= th for a in test))
    n, m = len(cal), len(test)
    law = S.late_count(n, m, 0)
    crit = next((q for q in range(m + 1) if law.sf(q - 1) <= 0.05), None)
    return {"n": n, "m": m, "late": x, "predicted_pass_odds": n / (n + m), "p_upper": S.late_tail(n, m, x),
            "refuted_from": crit, "expected": m / (n + 1)}


def number(a):
    return int("".join(ch for ch in a.name if ch.isdigit()))


def main():
    half = {"first 32 -> last 36": (lambda a: number(a) <= 32, lambda a: number(a) > 32)}
    isu_t = {"groups 1-29 -> 30-64": (lambda a: a.group <= 29, lambda a: a.group >= 30)}
    cond = {"conditions 1-2 -> 3": (lambda a: a.group <= 2, lambda a: a.group == 3)}
    fd = {"FD001 -> FD003": (lambda a: a.group == 1, lambda a: a.group == 3),
          "FD002 -> FD004": (lambda a: a.group == 2, lambda a: a.group == 4)}
    data = {"virkler": (virkler(), ("condition", "age"), half), "isu": (isu(), ("condition", "age"), isu_t),
            "cmapss": (turbofans(), ("condition", "age"), fd),
            "pronostia": (bearings("pronostia"), ("S1", "S2", "age"), cond),
            "xjtu": (bearings("xjtu"), ("S1", "S2", "age"), cond)}
    res = {f: analyse(*d) for f, d in data.items()}

    fam = [(f, s, nm) for f in res for s in res[f]["o5"] if s != "age" for nm in res[f]["o5"][s]]
    for key, p in zip(fam, holm([res[f]["o5"][s][nm]["p_upper"] for f, s, nm in fam])):
        res[key[0]]["o5"][key[1]][key[2]]["p_holm"] = float(p)

    sens = {}
    for nm, rows, kw in (("virkler own-life lead", virkler(), {"lead": "own"}), ("virkler double lead", virkler(), {"factor": 2.0}),
                         ("isu own-life lead", isu(), {"lead": "own"}), ("isu double lead", isu(), {"factor": 2.0}),
                         ("isu reference tests only", isu(daily=False), {}), ("isu end of life 0.85", isu(0.85), {}),
                         ("isu end of life 0.70", isu(0.70), {})):
        tabs = {s: Table(build(rows, s, **kw)) for s in ("condition", "age")}
        sens[nm] = {"assets": len(rows), **{s: float(t.loo().mean()) for s, t in tabs.items()},
                    "pair": summarise_pairs(tabs["condition"].loo() - tabs["age"].loo()),
                    "late_rank0": {s: against_law(sum(q["late"] for q in S.run_in_service(build(rows, s, **kw), 0)),
                                                  S.record_probs(len(rows), 0)) for s in tabs}}
    v, b = res["virkler"], res["isu"]
    inside = lambda o: o["range95"][0] <= o["late"] <= o["range95"][1]
    crit = {"C1": v["o2"]["condition_minus_age"]["mean"] < 0 and v["o2"]["condition_minus_age"]["p"] < 0.05,
            "C2": b["o2"]["condition_minus_age"]["mean"] < 0 and b["o2"]["condition_minus_age"]["p"] < 0.05,
            "C3": inside(v["signals"]["condition"]["o1"][0]), "C4": inside(v["signals"]["age"]["o1"][0])}
    (paths.RESULTS / "in_service.json").write_text(json.dumps(
        {"fleets": res, "sensitivity": sens, "criteria": crit}, indent=1, default=float), encoding="utf-8")

    for f, r in res.items():
        print(f"\n{f}: {r['assets']} assets, median life {r['median_life']:.1f} (range {r['life_range'][0]:.1f}-{r['life_range'][1]:.1f})")
        for s, o in r["signals"].items():
            q = o["o1"][0]
            print(f"  {s:<9} rank 0: late {q['late']} (expected {q['expected']:.2f}, range {q['range95']}, p {q['p_two_sided']:.3f}) at {q['late_at']}; "
                  f"LOO waste {o['loo_waste']:.3f} late {o['loo_late']} ties {o['ties']} inf {o['infinite']}")
            print("            waste vs n:", {n: round(x["rank0"], 3) for n, x in o["o3"].items()}, "level:", {n: round(x["level"], 3) for n, x in o["o3"].items()})
            print("            late by rank:", {k: (x["late"], round(x["expected"], 1), x["range95"]) for k, x in o["o1"].items()})
            c = {k: round((x["cost"][10][0] / x["cost"][10][1]) / o["run_to_failure_rate"][10], 3) for k, x in o["o1"].items()}
            print("            cost vs run-to-failure (r=10), recorded order:", c, "random order:", {k: round(x[10], 3) for k, x in o["random_order"].items()},
                  "fixed:", {n: (x["failures"], round((x["cost"][10][0] / x["cost"][10][1]) / o["run_to_failure_rate"][10], 3)) for n, x in o["fixed"].items()})
            print("            O6:", o["o6"])
        print("  O2:", r["o2"])
        for s, t in r["o5"].items():
            for nm, x in t.items():
                print(f"  O5 {s} {nm}: {x['late']}/{x['m']} late after n={x['n']}, odds {x['predicted_pass_odds']:.2f}, expected {x['expected']:.2f}, p {x['p_upper']:.4f}, refuted from {x['refuted_from']}, Holm {x.get('p_holm')}")
    print("\nsensitivity")
    for nm, x in sens.items():
        print(f"  {nm}: n {x['assets']} condition {x['condition']:.3f} age {x['age']:.3f} diff {x['pair']['mean']:+.3f} p {x['pair']['p']:.5f}; late {x['late_rank0']['condition']['late']} / {x['late_rank0']['age']['late']}")
    print("\ncriteria", crit)


if __name__ == "__main__":
    main()
