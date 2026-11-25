"""How much one calibration bearing controls the certificate (development screen 2).

PRONOSTIA development folds, seed 42, full process, cached CT-GRU inner models, in sample: the
threshold with and without the binding (largest lead-point) bearing. Writes results/dev_single_asset_control.json.
"""
import sys, json, math, collections
import numpy as np, torch
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
# (2) screen 2
def _refuse(*a, **k): raise RuntimeError("refusing to write inside the repository")
np.save = _refuse; torch.save = _refuse
from certalarm import assets, certify, crossfit, evaluate
from certalarm.predictors import modelcache
modelcache._save = _refuse
from certalarm.decision import outcome_curves
from certalarm.leadpoint import lead_point
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor
TH = np.asarray(evaluate.THETAS, float)
fleet = assets.load("pronostia"); by = {a.name: a for a in fleet}
pred = CTGRUPredictor("pronostia")
allcuts, foldrows = [], []
for fi, spec in enumerate(crossfit.build_fold(fleet, seed=42)):
    own = set(spec["test"]); cal = []
    for j, inner in enumerate(spec["inner"]):
        assert not set(inner["held"]) & own and not set(inner["fit"]) & own
        model = pred.fit([by[n] for n in inner["fit"]], seed=4200 + j, process="full")
        for name in inner["held"]:
            a = by[name]; mu, _ = pred.predict(model, a, np.ones(a.n_min, bool)); mu = np.asarray(mu, float)
            l, w = outcome_curves(mu, np.asarray(a.rul, float), TH, LEAD, np.ones(a.n_min, bool))
            cal.append((name, lead_point(mu, a.rul, LEAD), l, w, a.n_min))
    ts = np.array([c[1] for c in cal]); order = np.argsort(ts)[::-1]
    b = order[0]
    L = np.stack([c[2] for c in cal]); W = np.stack([c[3] for c in cal])
    th_all = certify.crc_threshold(L, TH, ALPHA); keep = [i for i in range(len(cal)) if i != b]
    th_wo = certify.crc_threshold(L[keep], TH, ALPHA)
    k_all, k_wo = int(np.searchsorted(TH, th_all)), int(np.searchsorted(TH, th_wo))
    w_all, w_wo = W[keep, k_all], W[keep, k_wo]
    cuts = 1 - w_wo / w_all; allcuts += list(cuts)
    foldrows.append((fi, len(cal), cal[b][0], cal[b][4], ts[order[0]], ts[order[1]], cal[order[1]][0], th_all, th_wo,
                     w_all.mean(), w_wo.mean(), 1 - w_wo.mean() / w_all.mean(), cuts.mean(), cuts.max(), (cuts > 0).sum(), len(cuts)))
for r in foldrows:
    print("fold %d n %d binding %s (%d min) t*1 %.4f t*2 %.4f (%s) th_all %.3f th_wo %.3f | waste others %.3f -> %.3f (fold-mean cut %.1f%%) | per-bearing cut mean %.1f%% max %.1f%% positive %d/%d" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], 100*r[11], 100*r[12], 100*r[13], r[14], r[15]))
print("pooled per-bearing cut mean %.2f%% max %.2f%%; mean of fold-mean cuts %.2f%%; mean of fold per-bearing-mean cuts %.2f%%" % (100*np.mean(allcuts), 100*np.max(allcuts), 100*np.mean([r[11] for r in foldrows]), 100*np.mean([r[12] for r in foldrows])))
wa = np.array([r[9] for r in foldrows]); ww = np.array([r[10] for r in foldrows])
print("exact: mean waste others all-cal %.5f without-binding %.5f ratio-of-means cut %.4f%%; bearings with lower waste %d of %d" % (wa.mean(), ww.mean(), 100 * (1 - ww.mean() / wa.mean()), sum(r[14] for r in foldrows), sum(r[15] for r in foldrows)))
json.dump({"folds": [list(map(lambda x: x if not isinstance(x, (np.floating, np.integer)) else x.item(), r)) for r in foldrows]}, open(REPO + "/results/dev_single_asset_control.json", "w"), indent=1)
