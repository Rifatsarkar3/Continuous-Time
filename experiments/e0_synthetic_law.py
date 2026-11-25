"""E0: realized late-alarm risk vs number of calibration assets on a synthetic fleet.

Hard gates (pre-registered in the Phase 1 plan):
  G_a  CRC realized risk <= alpha + 3 SE for every n (3 SE keeps the family-wise false-fail
       rate across the n values below ~1%)
  G_b  uncorrected empirical rule exceeds alpha at n = 2
  G_d  CRC is uncertifiable for every draw with n < min_assets(alpha) and for none with n >= it
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import baselines as B, certify as C, decision as D, sim as S  # noqa: E402
from certalarm.paths import RESULTS  # noqa: E402

ALPHA, LEAD, THETA0 = 0.10, 0.05, 0.15
THETAS = np.round(np.arange(0.0, 1.02, 0.01), 2)
N_LIST = (2, 4, 9, 19, 49, 99)
DRAWS = 2000
RULES = ("crc", "empirical", "point", "lbc", "supband")


def one_draw(n, d):
    fleet = S.simulate_fleet(n + 1, np.random.default_rng([2026, n, d]))
    cal, t = fleet[:n], fleet[n]
    Lm = np.stack([D.outcome_curves(u["mu"], u["rul"], THETAS, LEAD)[0] for u in cal])
    th_crc = C.crc_threshold(Lm, THETAS, ALPHA)
    th_emp = C.empirical_threshold(Lm, THETAS, ALPHA)
    q_lbc = B.lbc_quantile([u["mu"] for u in cal], [u["sigma"] for u in cal], [u["rul"] for u in cal], ALPHA)
    q_sup = B.supband_quantile([B.sup_score(u["mu"], u["sigma"], u["rul"]) for u in cal], ALPHA)
    rules = {
        "crc": (t["mu"], THETAS[-1] if th_crc is None else th_crc),
        "empirical": (t["mu"], THETAS[-1] if th_emp is None else th_emp),
        "point": (t["mu"], THETA0),
        "lbc": (B.lower_signal(t["mu"], t["sigma"], q_lbc), THETA0),
        "supband": (B.lower_signal(t["mu"], t["sigma"], q_sup), THETA0),
    }
    out = {k: D.alarm_outcome(sig, t["rul"], th, LEAD) for k, (sig, th) in rules.items()}
    out["uncertifiable"] = th_crc is None
    return out


def main():
    res = {}
    for n in N_LIST:
        draws = [one_draw(n, d) for d in range(DRAWS)]
        row = {"uncertifiable_frac": float(np.mean([x["uncertifiable"] for x in draws]))}
        for k in RULES:
            late = np.array([x[k][0] for x in draws], float)
            rem = np.array([x[k][1] for x in draws], float)
            row[k] = {"risk": float(late.mean()), "se": float(late.std(ddof=1) / np.sqrt(DRAWS)),
                      "waste": float(rem.mean())}
        res[str(n)] = row
        print(n, {k: round(row[k]["risk"], 3) for k in RULES}, "uncert", row["uncertifiable_frac"])
    n_min = C.min_assets(ALPHA)
    gates = {
        "G_a_crc_valid_all_n": all(res[str(n)]["crc"]["risk"] <= ALPHA + 3 * max(res[str(n)]["crc"]["se"], 1e-9)
                                   for n in N_LIST),
        "G_b_empirical_exceeds_at_n2": res["2"]["empirical"]["risk"] > ALPHA,
        "G_d_floor_exact": all((res[str(n)]["uncertifiable_frac"] == 1.0) if n < n_min else
                               (res[str(n)]["uncertifiable_frac"] == 0.0) for n in N_LIST),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e0_synthetic_law.json").write_text(json.dumps({"alpha": ALPHA, "lead": LEAD, "draws": DRAWS,
                                                               "results": res, "gates": gates}, indent=2))
    lines = ["# E0: asset-count law on a synthetic heterogeneous fleet", "",
             f"alpha = {ALPHA}, lead = {LEAD}, {DRAWS} draws per n, min certifiable n = {n_min}", "",
             "| n | " + " | ".join(f"{k} risk" for k in RULES) + " | CRC waste | LBC waste | uncertifiable |",
             "|" + "---|" * (len(RULES) + 4)]
    for n in N_LIST:
        r = res[str(n)]
        lines.append(f"| {n} | " + " | ".join(f"{r[k]['risk']:.3f}" for k in RULES)
                     + f" | {r['crc']['waste']:.3f} | {r['lbc']['waste']:.3f} | {r['uncertifiable_frac']:.2f} |")
    lines += ["", "Gates: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items())]
    (RESULTS / "e0_synthetic_law.md").write_text("\n".join(lines) + "\n")
    print("GATES", gates)
    print("PHASE-1 E0 GATE:", "PASS" if all(gates.values()) else "FAIL")


if __name__ == "__main__":
    main()
