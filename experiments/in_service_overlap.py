"""Certification in service with several assets in service at a time, recorded order. Post hoc, descriptive.

Added after the pre-declared run of in_service_certification.py (same fleets, signals and leads) to show what the
worst-case law for overlapping service, P(late at the j-th entry) <= (k + s) / j with s assets in service, costs in
practice. Threshold fixed at entry from the assets completed by then.

    python experiments/in_service_overlap.py        # writes results/in_service_overlap.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import paths, service as S  # noqa: E402
from experiments import in_service_certification as X  # noqa: E402

SLOTS = (1, 4, 16)


def main():
    data = {"virkler": (X.virkler(), ("condition", "age")), "isu": (X.isu(), ("condition", "age")),
            "cmapss": (X.turbofans(), ("condition", "age")), "pronostia": (X.bearings("pronostia"), ("S1", "S2", "age")),
            "xjtu": (X.bearings("xjtu"), ("S1", "S2", "age"))}
    res = {}
    for f, (rows, signals) in data.items():
        N = len(rows)
        res[f] = {"assets": N, "signals": {}}
        for sig in signals:
            fleet = X.build(rows, sig)
            out = {}
            for s in SLOTS:
                if s >= N:
                    continue
                runs = S.run_overlapping(fleet, s)
                late = [r["late"] for r in runs]
                for j, r in enumerate(runs):                       # the worst-case law, path by path
                    if r["late"]:
                        assert sum(a.lead_value > fleet[j].lead_value for a in fleet[: j + 1]) < s
                out[s] = {"late": int(sum(late)), "bound": float(S.record_probs(N, s - 1).sum()),
                          "without_alarm": int(sum(r["theta"] == -np.inf for r in runs)),
                          "mean_waste": float(np.mean([r["waste"] for r in runs if r["theta"] > -np.inf]))}
            res[f]["signals"][sig] = out
            print(f, N, sig, {s: (o["late"], o["without_alarm"], round(o["bound"], 1), round(o["mean_waste"], 3)) for s, o in out.items()})
    (paths.RESULTS / "in_service_overlap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
