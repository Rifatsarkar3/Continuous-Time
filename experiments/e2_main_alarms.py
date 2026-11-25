"""E2: late-alarm rate and wasted life of every rule, both bearing datasets, three deployments.

Smoke first:
    python experiments/e2_main_alarms.py --smoke
Full:
    python experiments/e2_main_alarms.py --seeds 42 123 777 999 7 --dsaflite --dann
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, checkpoint, crossfit, evaluate, features, paths, registry  # noqa: E402
from certalarm.predictors import ctgru_train, dsaf_train, modelcache  # noqa: E402
from certalarm.predictors.ctgru import CTGRU  # noqa: E402
from certalarm.predictors.dsaflite import DSAFLite  # noqa: E402

ALPHA, LEAD = 0.10, 0.05
PROCESSES = ("full", "mixture", "sparse_shift")


class CTGRUPredictor:
    """Minute-grid features; the observation mask feeds the model and gates the alarm."""

    def __init__(self, dataset, cache=True, late_weight=0.0, asym=1.0, name=None, iters=None,
                 gate_asym=False):
        self.dataset = dataset
        self.cache = cache
        self.late_weight, self.asym, self.gate_asym = float(late_weight), float(asym), bool(gate_asym)
        self.name = name or "ctgru"
        self.iters = iters or ctgru_train.ITERS[dataset]
        self.grids = {}   # raw, un-normalized grids; normalization stays fold-local

    def grid(self, asset):
        if asset.name not in self.grids:
            self.grids[asset.name] = features.asset_grid(
                asset.dataset, asset.name, asset.windows, asset.per_min,
                cache_dir=paths.DATA / "features")
        return self.grids[asset.name]

    def fit(self, fit_assets, seed, process):
        # The scale is deterministic in the fit bearings, so it is recomputed rather than cached
        # and always matches whichever weights come back.
        scale = features.fit_scale([self.grid(a) for a in fit_assets])
        X = [features.normalize(self.grid(a), scale) for a in fit_assets]
        r = [a.rul for a in fit_assets]
        # CT-GRU trains under the observation process, so the process belongs in the key.
        extra = {"process": process, "iters": self.iters}
        # The loss settings change the weights, so they belong in the key - but only when they
        # differ from the original loss, so every model already cached stays a valid hit.
        if (self.late_weight, self.asym) != (0.0, 1.0):
            extra["loss"] = {"late_weight": self.late_weight, "asym": self.asym}
            if self.gate_asym:
                extra["loss"]["gate_asym"] = True
        k = modelcache.key("ctgru", self.dataset, seed, [a.name for a in fit_assets], extra)
        model = modelcache.cached(
            k, lambda: CTGRU(X[0].shape[1]),
            lambda: ctgru_train.train(X, r, process, self.iters, seed,
                                      late_weight=self.late_weight, asym=self.asym,
                                      gate_asym=self.gate_asym),
            enabled=self.cache)
        return {"model": model, "scale": scale}

    def predict(self, model, asset, M):
        X = features.normalize(self.grid(asset), model["scale"])
        return ctgru_train.predict(model["model"], X, M.astype(np.float32))


class DSAFLitePredictor:
    """Raw windows; the observation mask gates the alarm only, since the model is per-window."""

    def __init__(self, dann=False, epochs=None, cache=True):
        self.dann, self.epochs, self.cache = dann, epochs, cache
        self.name = "dsaflite_dann" if dann else "dsaflite"
        self._cache = {}

    def fit(self, fit_assets, seed, process):
        # DSAFLite never reads the observation mask while training, so a model depends only on
        # its fit bearings and its seed - not on the deployment process. Reusing it across the
        # three processes is exact deduplication, not an approximation, and it cuts this arm
        # from 300 fits per dataset to 100. The key carries the exact fit-bearing set, so a
        # model can never be reused for a fold where it would have seen a forbidden bearing.
        names = tuple(sorted(a.name for a in fit_assets))
        if (seed, names) not in self._cache:
            k = modelcache.key(self.name, fit_assets[0].dataset, seed, names,
                               {"dann": self.dann, "epochs": self.epochs})
            self._cache[(seed, names)] = modelcache.cached(
                k, lambda: DSAFLite(feat_dim=dsaf_train.RECIPE["feat_dim"],
                                    dropout=dsaf_train.RECIPE["dropout"]),
                lambda: dsaf_train.train(fit_assets, seed=seed, epochs=self.epochs,
                                         dann=self.dann), enabled=self.cache)
        return {"model": self._cache[(seed, names)]}

    def predict(self, model, asset, M):
        return dsaf_train.predict_minutes(model["model"], asset)


def time_one_model(predictor, fleet, fold, seed):
    t0 = time.perf_counter()
    by = {a.name: a for a in fleet}
    predictor.fit([by[n] for n in fold["inner"][0]["fit"]], seed=seed, process="full")
    return time.perf_counter() - t0


def run(dataset, predictor, seeds, processes, n_rep, use_registry=True, ckpt_tag=None):
    fleet = assets.load(dataset)
    done = checkpoint.completed(ckpt_tag) if ckpt_tag else {}
    rows, reused = [], 0
    for seed in seeds:
        specs = crossfit.build_fold(fleet, seed=seed)
        for process in processes:
            for i, spec in enumerate(specs):
                unit = {"dataset": dataset, "predictor": predictor.name, "process": process,
                        "seed": seed, "fold": i}
                ukey = checkpoint.unit_key(unit)
                if ukey in done:
                    # Already finished, and its one permitted test evaluation is already spent.
                    rows += done[ukey]
                    reused += 1
                    continue
                fold = crossfit.run_fold(fleet, spec, predictor, process, n_rep, seed)
                rules = evaluate.fit_rules(fold.calibration, ALPHA, LEAD)
                tag = f"{dataset}/{predictor.name}/{process}/seed{seed}/fold{i}"
                if use_registry:
                    cfg = {"tag": tag, "alpha": ALPHA, "lead": LEAD, "n_rep": n_rep,
                           "test": spec["test"],
                           "rules": {k: v["theta"] for k, v in rules.items()}}
                    for m in fold.models:
                        registry.register(cfg, m["model"])
                        registry.check(cfg, m["model"], tag)
                got = evaluate.apply_rules(rules, fold.test, LEAD)
                # Predictor accuracy on the very bearings the alarm was judged on. Wasted life
                # is only interpretable against it: a rule that gives up most of the remaining
                # life may be paying for a weak predictor rather than for the guarantee itself.
                mae = {n: float(np.mean([np.abs(t.mu - t.rul).mean() for t in reps]))
                       for n, reps in fold.test.items()}
                unit_rows = [{"dataset": dataset, "predictor": predictor.name,
                              "process": process, "seed": seed, "fold": i,
                              "rule": rule, "asset": asset_name,
                              "certifiable": rules[rule].get("certifiable", False),
                              "theta": rules[rule]["theta"],
                              "predictor_mae": mae[asset_name], **row}
                             for rule, per_asset in got.items()
                             for asset_name, row in per_asset.items()]
                if ckpt_tag:
                    checkpoint.append(ckpt_tag, unit, unit_rows)
                rows += unit_rows
                print(f"  [done] {ukey}", flush=True)
    if reused:
        print(f"  [resume] reused {reused} finished units from the checkpoint", flush=True)
    return rows


def report(rows):
    for pred in sorted({r["predictor"] for r in rows}):
        sel = [r for r in rows if r["predictor"] == pred]
        mae = np.mean([r["predictor_mae"] for r in sel])
        print(f"\n=== {pred} on {sorted({r['dataset'] for r in sel})} "
              f"| predictor MAE on test bearings {mae:.4f} ===")
        _rule_table(sel)


def _rule_table(rows):
    print(f"{'rule':<14}{'late':>8}{'upper':>8}{'waste':>8}{'assets':>8}   certifiable")
    for rule in evaluate.RULES:
        sel = [r for r in rows if r["rule"] == rule]
        if not sel:
            continue
        # Every field that distinguishes a row belongs in the key. `report` pools all three
        # observation processes, so leaving `process` out collides each row with its twin from
        # the other processes and keeps only the last one.
        keyed = {f"{r['process']}/{r['seed']}/{r['fold']}/{r['asset']}": r for r in sel}
        # Cluster on the bearing: the seeds of one bearing are not independent observations.
        s = evaluate.summarize(keyed, cluster_key={k: r["asset"] for k, r in keyed.items()})
        cert = all(r["certifiable"] for r in sel)
        print(f"{rule:<14}{s['late_rate']:>8.3f}{s['late_upper']:>8.3f}"
              f"{s['waste_mean']:>8.3f}{s['n_assets']:>8}   {cert}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one seed, PRONOSTIA, CT-GRU only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777, 999, 7])
    ap.add_argument("--datasets", nargs="+", default=["pronostia", "xjtu"])
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--processes", nargs="+", default=list(PROCESSES),
                    help="observation processes to deploy under; fewer means a cheaper run and "
                         "no E6 evidence for the ones left out")
    ap.add_argument("--dsaflite", action="store_true", help="also run the DSAFLite predictor")
    ap.add_argument("--dann", action="store_true", help="also run the DANN-on ablation arm")
    ap.add_argument("--no-ctgru", action="store_true", help="skip the CT-GRU predictor")
    ap.add_argument("--tag", default="", help="suffix for the result file, so long runs can be "
                                              "split into pieces that fail independently")
    a = ap.parse_args()

    if a.smoke:
        a.seeds, a.datasets, a.n_rep = [42], ["pronostia"], 4
        a.dsaflite = a.dann = False
    unknown = [p for p in a.processes if p not in PROCESSES]
    if unknown:
        raise SystemExit(f"unknown observation process(es): {unknown}; known: {list(PROCESSES)}")

    rows = []
    for dataset in a.datasets:
        preds = [] if a.no_ctgru else [CTGRUPredictor(dataset)]
        if a.dsaflite:
            preds.append(DSAFLitePredictor(dann=False))
        if a.dann:
            preds.append(DSAFLitePredictor(dann=True))
        for p in preds:
            fleet = assets.load(dataset)
            spec = crossfit.build_fold(fleet, seed=a.seeds[0])[0]
            secs = time_one_model(p, fleet, spec, a.seeds[0])
            n_models = len(a.seeds) * len(a.processes) * 5 * 4
            print(f"[timing] {dataset}/{p.name}: one inner model {secs:.1f}s; "
                  f"{n_models} models in the grid is about {secs * n_models / 3600:.2f} h",
                  flush=True)
            rows += run(dataset, p, a.seeds, tuple(a.processes), a.n_rep,
                        use_registry=not a.smoke,
                        ckpt_tag=None if a.smoke else f"{a.tag or 'run'}_{dataset}_{p.name}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                          .isoformat(timespec="seconds"),
                          "git_commit": commit, "smoke": a.smoke, "seeds": a.seeds,
                          "n_rep": a.n_rep, "alpha": ALPHA, "lead": LEAD},
           "rows": rows}
    paths.RESULTS.mkdir(exist_ok=True)
    name = ("e2_main_alarms_smoke.json" if a.smoke
            else f"e2_main_alarms{'_' + a.tag if a.tag else ''}.json")
    (paths.RESULTS / name).write_text(json.dumps(out, indent=2))
    report(rows)
    print(f"\nwritten to results/{name}")


if __name__ == "__main__":
    main()
