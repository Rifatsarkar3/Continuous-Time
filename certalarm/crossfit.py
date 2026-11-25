"""Nested cross-fitting over assets.

For one outer fold: train one model per inner fold on that inner fold's fit assets, and predict
each inner fold's held assets with the model that did not see them. Every calibration asset
therefore carries a trajectory from a model blind to it, which is what makes the calibration
losses exchangeable with a new asset's loss - the condition the certification guarantee rests on.

A test asset is scored under each inner model separately and its **outcomes** are averaged; the
**predictions** are never averaged. The distinction decides whether the guarantee holds. An
averaged prediction is smoother than any single model's, and because the alarm fires at the
first crossing of a running minimum, a smoother signal crosses later. Calibration assets, scored
by one model each, would then look less late than the smoother deployed signal really is, the
calibrated threshold would come out too aggressive, and the realized late rate could exceed the
promised risk. Averaging outcomes instead leaves every trajectory a single-model trajectory,
exchangeable with the calibration ones, and leaves the expected loss - the quantity conformal
risk control actually bounds - unchanged. This is the CV+ / CV-CRC construction.

Observation replicates are drawn once per asset and shared by every rule, so rules are compared
on identical deployments.
"""
from dataclasses import dataclass, field

import numpy as np

from . import folds, observe, pooling


@dataclass
class Trajectory:
    mu: np.ndarray        # (n_min,) predicted normalized remaining life
    sigma: np.ndarray     # (n_min,) predictive scale, strictly positive
    rul: np.ndarray       # (n_min,) true normalized remaining life
    can_fire: np.ndarray  # (n_min,) bool, minutes carrying an acquired measurement
    scale: float = 1.0    # this trajectory's condition scale, from ITS model's fit assets only


@dataclass
class FoldResult:
    fold: dict
    models: list
    calibration: dict = field(default_factory=dict)          # name -> [Trajectory]
    test: dict = field(default_factory=dict)                 # name -> [Trajectory]
    calibration_sources: dict = field(default_factory=dict)  # name -> the model that predicted it


def build_fold(assets, n_outer=5, n_inner=4, seed=0):
    return folds.nested_folds(assets, n_outer=n_outer, n_inner=n_inner, seed=seed)


def _masks(assets, process, n_rep, seed):
    return dict(zip([a.name for a in assets],
                    observe.draw(process, [a.rul for a in assets],
                                 [a.name for a in assets], n_rep, seed)))


def run_fold(assets, fold, predictor, process, n_rep, seed, lead=0.05):
    """Fit the inner models of one outer fold and produce every trajectory it needs."""
    by_name = {a.name: a for a in assets}
    masks = _masks(assets, process, n_rep, seed)
    res = FoldResult(fold=fold, models=[])

    scales_of = []
    for j, inner in enumerate(fold["inner"]):
        fit_assets = [by_name[n] for n in inner["fit"]]
        model = predictor.fit(fit_assets, seed=seed * 100 + j, process=process)
        res.models.append(model)
        # Scales come from this inner model's OWN fit assets, never from the assets it will be
        # used to score. That is what keeps calibration and test losses exchangeable; drawing
        # them from the calibration set would make every calibration loss depend on the whole
        # calibration set and quietly void the guarantee.
        scales = pooling.condition_scales(
            {a.name: {"condition": a.condition,
                      "reps": [_trajectory(predictor, model, a, masks[a.name][0])]}
             for a in fit_assets},
            lead, fit_names=[a.name for a in fit_assets])
        scales_of.append(scales)
        for name in inner["held"]:
            res.calibration_sources[name] = model
            res.calibration[name] = [
                _trajectory(predictor, model, by_name[name], m,
                            scales.get(by_name[name].condition, 1.0)) for m in masks[name]]

    for name in fold["test"]:
        asset = by_name[name]
        res.test[name] = [_trajectory(predictor, model, asset, m,
                                      scales_of[k].get(asset.condition, 1.0))
                          for m in masks[name] for k, model in enumerate(res.models)]
    return res


def _trajectory(predictor, model, asset, mask, scale=1.0):
    M = mask["M"]
    mu, sigma = predictor.predict(model, asset, M)
    return Trajectory(np.asarray(mu, float), np.asarray(sigma, float),
                      asset.rul.astype(float), M.astype(bool), float(scale))


def deployed_rule_description():
    """The rule whose risk is certified: draw one inner model uniformly, then alarm with it.

    A test asset is scored under every inner model and the resulting outcomes are averaged, which
    is the risk of picking one of them at random. That is what keeps the construction honest.
    """
    return "uniform mixture over the inner cross-fitted models"
