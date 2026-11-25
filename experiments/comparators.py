"""Parametric tolerance limits and uncertified cost-optimal thresholds next to the certified rules, on four fleets.

Descriptive: the fleets' lead-point values and certified outcomes were known. Plan de232f5 (2026-11-25): rules R0, L, LN,
W, C and the reference O; n = 9 and 36; 2,000 calibration sets without replacement, the same for every rule; all other
assets test. The keys "charged", "routes", "overlap_cost", "held_cost", "hindsight_by_group" and "training_charged" are post
hoc, added after two reviews: the cost of the whole fleet with the run-to-failure calibration assets charged like field
failures; the same with a bench failure at one replacement and no service, and the bench price at which a route breaks even
with the in-service rule; nine bench failures followed by the in-service rule; costs with several assets in service; the
in-service cost of the ISU cells with the last record held; the best age in hindsight per batch, policy, duty or subset; and
the in-service cost of the turbofans with the units that trained the ridge charged.

    python experiments/comparators.py        # writes results/comparators.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import paths, service as S, tolerance as TL  # noqa: E402
from experiments import in_service_certification as X  # noqa: E402
from certalarm.data import fleets  # noqa: E402
from experiments.in_service_matr import LEAD_CYCLES, held, matr_rows  # noqa: E402

ALPHA, CONTENT, CONFIDENCE, DRAWS, SIZES, COSTS = 0.10, 0.90, 0.975, 2000, (9, 36), (10, 5, 20)


def quantity(values, kind):
    """-> (lead-point quantity, is the limit an upper one); late = at or beyond the limit."""
    return (-values, False) if kind in ("age", "crack") else (values, True)


def limits(values, idx, kind):
    """{rule: threshold per row of idx} of the parametric rules."""
    v, upper = quantity(values, kind)
    x = v[idx]
    if kind == "ridge":
        out = {"LN": TL.normal_limit(x, CONTENT, CONFIDENCE, upper)}
    else:
        assert (v > 0).all(), kind
        out = {"LN": TL.lognormal_limit(x, CONTENT, CONFIDENCE, upper), "W": TL.weibull_limit(x, CONTENT, CONFIDENCE, upper)}
    return {k: (t if upper else -t) for k, t in out.items()}


def cheapest(late, time, T, r):
    """Column with the lowest cost rate, or -1 for no alarm. late, time: assets x candidate thresholds."""
    rate = (r * late + (1 - late)).sum(0) / time.sum(0)
    j = int(np.argmin(rate))
    return j if rate[j] < r * len(T) / T.sum() else -1


def score(late, waste, time, T, test):
    """Late fraction, waste and cost relative to running to failure, draw by draw. late etc.: assets x draws."""
    m = test.sum(0)
    out = {"late": (late * test).sum(0) / m, "waste": (waste * test).sum(0) / m}
    for r in COSTS:
        cost = ((r * late + (1 - late)) * test).sum(0) / (time * test).sum(0)
        out[f"cost{r}"] = cost / (r * m / (T[:, None] * test).sum(0))
    return out


def summary(s):
    return {"late": float(s["late"].mean()), "exceeds": float(np.mean(s["late"] > ALPHA + 1e-12)), "waste": float(s["waste"].mean()),
            **{k: float(v.mean()) for k, v in s.items() if k.startswith("cost")}}


def study(fleet, kind):
    tab = X.Table(fleet)
    values, T, N = tab.values, tab.T, tab.N
    assert np.isfinite(values).all()
    never = {"late": np.ones(N), "waste": np.zeros(N), "time": T}
    out = {"assets": N}
    for n in SIZES:
        rng = np.random.default_rng(0)
        idx = np.array([rng.permutation(N)[:n] for _ in range(DRAWS)])
        test = np.ones((N, DRAWS))
        test[idx, np.arange(DRAWS)[:, None]] = 0.0
        res = {}
        desc = np.take_along_axis(idx, np.argsort(-values[idx], axis=1, kind="stable"), axis=1)     # indices, largest value first
        for rule, k in (("R0", 0), ("L", max(0, S.level_rank(n, ALPHA)))):
            j = desc[:, k]
            res[rule] = score(tab.late[:, j], tab.waste[:, j], tab.time[:, j], T, test)
        for rule, th in limits(values, idx, kind).items():
            t = X.Table(fleet, th)
            res[rule] = score(t.late, t.waste, t.time, T, test)
        for r in COSTS:                                                    # C: lowest observed cost on the calibration records
            pick = [(c, cheapest(tab.late[np.ix_(c, c)], tab.time[np.ix_(c, c)], T[c], r)) for c in idx]
            cols = {q: np.column_stack([never[q] if j < 0 else getattr(tab, q)[:, c[j]] for c, j in pick]) for q in never}
            s = score(cols["late"], cols["waste"], cols["time"], T, test)
            res[f"C{r}"] = {**{k: s[k] for k in ("late", "waste")}, f"cost{r}": s[f"cost{r}"], "no_alarm": np.array([j < 0 for _, j in pick], float)}
        out[str(n)] = {rule: summary(s) | ({"no_alarm": float(s["no_alarm"].mean())} if "no_alarm" in s else {}) for rule, s in res.items()}
    out["oracle"] = {}
    for r in COSTS:
        j = cheapest(tab.late, tab.time, T, r)
        late, time = (np.ones(N), T) if j < 0 else (tab.late[:, j], tab.time[:, j])
        out["oracle"][r] = {"late": float(late.mean()), "cost": float((r * late + 1 - late).sum() / time.sum() / (r * N / T.sum())),
                            "rank": -1 if j < 0 else int((values > values[j]).sum())}
    return out


def charged(fleet, kind):
    """Cost of the whole fleet when n assets are run to failure first and the rule then serves the rest; same draws as `study`."""
    tab = X.Table(fleet)
    v, T, N = tab.values, tab.T, tab.N
    out = {}
    for n in SIZES:
        rng = np.random.default_rng(0)
        idx = np.array([rng.permutation(N)[:n] for _ in range(DRAWS)])
        test = np.ones((N, DRAWS))
        test[idx, np.arange(DRAWS)[:, None]] = 0.0
        desc = np.take_along_axis(idx, np.argsort(-v[idx], axis=1, kind="stable"), axis=1)
        rules = {"R0": (tab.late[:, desc[:, 0]], tab.time[:, desc[:, 0]])}
        for k, th in limits(v, idx, kind).items():
            t = X.Table(fleet, th)
            rules[k] = (t.late, t.time)
        time0 = (T[:, None] * (1 - test)).sum(0)
        out[str(n)] = {k: {r: float(np.mean((((r * late + 1 - late) * test).sum(0) + r * n) / ((time * test).sum(0) + time0)) * T.sum() / (r * N))
                           for r in COSTS} for k, (late, time) in rules.items()}
    return out


def routes(fleet, kind, r=10, n0=9):
    """Routes at cost ratio r on 2,000 random orders. in_service: rank 0 from the first asset. lognormal9, rank0_36: calibrate
    on the bench, then keep the threshold. bench9_in_service: nine bench failures, then rank 0 in service. Each bench route with
    its failures charged r and their time credited (field), and charged one replacement with no service (bench)."""
    tab = X.Table(fleet)
    v, T, N = tab.values, tab.T, tab.N
    base = r * N / T.sum()
    rng = np.random.default_rng(0)
    perms = [rng.permutation(N) for _ in range(DRAWS)]
    serve = float(np.mean([((r - 1) * f + N) / tm for f, tm in (X.permuted_cost(tab, q, 0) for q in perms)]) / base)
    out = {"in_service": serve}

    def price(field, service_cost, service_time, n, cal_time):
        bench_cost = (service_cost + (r if field else 1) * n) / (service_time + (cal_time if field else 0.0))
        return bench_cost / base

    hyb = {"field": [], "bench": []}
    for q in perms:
        top, cost, time = q[:n0][np.argmax(v[q[:n0]])], 0.0, 0.0
        for i in q[n0:]:
            late = bool(tab.late[i, top])
            cost, time = cost + (r if late else 1), time + tab.time[i, top]
            if late:
                top = i
        for k in hyb:
            hyb[k].append(price(k == "field", cost, time, n0, T[q[:n0]].sum()))
    out["bench9_in_service"] = {k: float(np.mean(x)) for k, x in hyb.items()}
    for name, n, rule in (("lognormal9", 9, "LN"), ("rank0_36", 36, "R0")):
        rng = np.random.default_rng(0)
        idx = np.array([rng.permutation(N)[:n] for _ in range(DRAWS)])
        test = np.ones((N, DRAWS))
        test[idx, np.arange(DRAWS)[:, None]] = 0.0
        if rule == "LN":
            t = X.Table(fleet, limits(v, idx, kind)["LN"])
            late, time = t.late, t.time
        else:
            j = idx[np.arange(DRAWS), np.argmax(v[idx], axis=1)]
            late, time = tab.late[:, j], tab.time[:, j]
        cost, tm = ((r * late + 1 - late) * test).sum(0), (time * test).sum(0)
        out[name] = {"field": float(np.mean(price(True, cost, tm, n, (T[:, None] * (1 - test)).sum(0)))), "bench": float(np.mean(price(False, cost, tm, n, 0.0))),
                     "break_even_bench_price": float((serve * base - np.mean(cost / tm)) / np.mean(n / tm))}
    return out


def overlap_cost(fleet):
    T = np.array([a.T for a in fleet])
    out = {}
    for slots in (1, 4, 16):
        run = S.run_overlapping(fleet, slots)
        out[slots] = {r: float((c / t) / (r * len(T) / T.sum())) for r in COSTS for c, t in [X.campaign(run, r)]}
    return out


def held_cost():
    """In-service cost of the ISU cells, rank 0, recorded order, with the last recorded capacity held between tests."""
    cells = fleets.isu_ilcc()
    fleet = X.build(held(X.isu(), lambda n: (cells[n][0], cells[n][1] / cells[n][1][0])), "condition")
    run, T = S.run_in_service(fleet, 0), np.array([a.T for a in fleet])
    out = {"late": int(sum(q["late"] for q in run))}
    for r in COSTS:
        c, t = X.campaign(run, r)
        out[r] = float((c / t) / (r * len(T) / T.sum()))
    return out


def hindsight(fleet, groups):
    """Cost of the cheapest threshold of each group, chosen on the group itself."""
    tab, groups = X.Table(fleet), np.array(groups)
    out = {}
    for r in COSTS:
        cost = time = 0.0
        for g in np.unique(groups):
            ix = np.flatnonzero(groups == g)
            j = cheapest(tab.late[np.ix_(ix, ix)], tab.time[np.ix_(ix, ix)], tab.T[ix], r)
            late, tm = (np.ones(len(ix)), tab.T[ix]) if j < 0 else (tab.late[ix, ix[j]], tab.time[ix, ix[j]])
            cost, time = cost + (r * late + 1 - late).sum(), time + tm.sum()
        out[r] = float((cost / time) / (r * tab.N / tab.T.sum()))
    return out


def training_charged():
    """In-service cost of the turbofans, rank 0, recorded order, with the failures of the units that trained the ridge."""
    train = X.partition(X.cmapss.load(), 42)[0]
    lives = np.array([float(u.n_min - 1) for u in train])
    fleet = X.build(X.turbofans(), "condition")
    run, T = S.run_in_service(fleet, 0), np.array([a.T for a in fleet])
    out = {"training_units": len(lives)}
    for r in COSTS:
        cost, time = X.campaign(run, r)
        out[r] = float(((cost + r * len(lives)) / (time + lives.sum())) / (r * (len(T) + len(lives)) / (T.sum() + lives.sum())))
    return out


def transfer(fleet, kind, in_cal, in_test):
    cal, test = [a for a in fleet if in_cal(a)], [a for a in fleet if in_test(a)]
    values = np.array([a.lead_value for a in cal])
    tab = X.Table(cal)
    th = {"R0": S.threshold(values, 0)} | {k: float(t[0]) for k, t in limits(values, np.arange(len(cal))[None], kind).items()}
    j = cheapest(tab.late, tab.time, tab.T, 10)
    th["C10"] = -np.inf if j < 0 else float(np.nextafter(values[j], np.inf))
    return {"n": len(cal), "m": len(test), **{k: int(sum(a.lead_value >= t for a in test)) for k, t in th.items()}}


def main():
    half = {"first 32 -> last 36": (lambda a: X.number(a) <= 32, lambda a: X.number(a) > 32)}
    batch = {"batch 1 -> batch 2": (lambda a: a.group == 1, lambda a: a.group == 2),
             "batches 1-2 -> batch 3": (lambda a: a.group <= 2, lambda a: a.group == 3)}
    isu_t = {"groups 1-29 -> 30-64": (lambda a: a.group <= 29, lambda a: a.group >= 30)}
    fd = {"FD001 -> FD003": (lambda a: a.group == 1, lambda a: a.group == 3),
          "FD002 -> FD004": (lambda a: a.group == 2, lambda a: a.group == 4)}
    cells, policy = matr_rows()
    data = {"virkler": (X.virkler(), {}, "crack", half), "matr": (cells, {"lead": LEAD_CYCLES}, "capacity", batch),
            "isu": (X.isu(), {}, "capacity", isu_t), "cmapss": (X.turbofans(), {}, "ridge", fd)}
    res = {}
    for f, (rows, kw, kind, trans) in data.items():
        res[f] = {}
        for sig in ("condition", "age"):
            fleet = X.build(rows, sig, **kw)
            k = kind if sig == "condition" else "age"
            res[f][sig] = {**study(fleet, k), "transfer": {nm: transfer(fleet, k, *g) for nm, g in trans.items()}}
            res[f][sig]["charged"] = charged(fleet, k)
            res[f][sig]["overlap_cost"] = overlap_cost(fleet)
            if sig == "condition":
                res[f][sig]["routes"] = {r: routes(fleet, k, r) for r in COSTS}
            if sig == "age":
                res[f][sig]["hindsight_by_group"] = {"group": hindsight(fleet, [a.group for a in fleet])}
                if f == "matr":
                    res[f][sig]["hindsight_by_group"]["policy"] = hindsight(fleet, [policy[a.name] for a in fleet])
            print(f"\n{f} {sig}")
            for n in SIZES:
                for rule, s in res[f][sig][str(n)].items():
                    print(f"  n={n:<3} {rule:<4} late {s['late']:.3f}  exceeds {s['exceeds']:.3f}  waste {s['waste']:.3f}  "
                          f"cost {s.get('cost10', s.get('cost5', s.get('cost20'))):.3f}", {k: round(v, 3) for k, v in s.items() if k == "no_alarm"})
            print("  oracle", res[f][sig]["oracle"], "\n  transfer", res[f][sig]["transfer"])
    res["cmapss"]["condition"]["training_charged"] = training_charged()
    res["isu"]["condition"]["held_cost"] = held_cost()
    (paths.RESULTS / "comparators.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
