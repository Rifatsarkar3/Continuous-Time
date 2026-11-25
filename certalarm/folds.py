"""Nested, condition-stratified fold assignment over assets.

Outer folds choose the test bearings. Inner folds cross-fit the remainder, so every calibration
bearing is predicted by a model that never trained on it. Both levels are stratified by operating
condition: an unstratified fold could hold out an entire condition and then measure transfer
rather than certification.
"""
import numpy as np


def _stratified_groups(names, condition_of, k, rng):
    """Deal the assets of each condition round-robin into k groups.

    The slot counter carries across conditions, which keeps the groups within one asset of each
    other in size. Its starting offset and the order the conditions are visited are both drawn
    from `rng`, and that part is not cosmetic: with a fixed offset and a fixed order, a condition
    with fewer than k assets lands in the same groups for every seed. On PRONOSTIA (7, 7 and 3
    assets over 3 conditions, 5 folds) the three condition-3 bearings always started at slot 14,
    so two of the five folds could never test that condition no matter how the run was re-seeded.
    """
    by_cond = {}
    for n in names:
        by_cond.setdefault(condition_of(n), []).append(n)
    groups = [[] for _ in range(k)]
    order = sorted(by_cond)
    rng.shuffle(order)
    slot = int(rng.integers(k))
    for cond in order:
        members = sorted(by_cond[cond])
        rng.shuffle(members)
        for m in members:
            groups[slot % k].append(m)
            slot += 1
    return [sorted(g) for g in groups]


def nested_folds(assets, n_outer=5, n_inner=4, seed=0):
    """-> list of {test, calibration, inner: [{fit, held}, ...]}, one entry per outer fold."""
    names = sorted(a.name for a in assets)
    if len(names) < n_outer:
        raise ValueError(f"need at least {n_outer} assets for {n_outer} outer folds, "
                         f"got {len(names)}")
    cond = {a.name: a.condition for a in assets}
    rng = np.random.default_rng(seed)
    out = []
    for test in _stratified_groups(names, cond.__getitem__, n_outer, rng):
        calibration = sorted(n for n in names if n not in test)
        if len(calibration) < n_inner:
            raise ValueError(f"need at least {n_inner} non-test assets for {n_inner} inner folds")
        inner = [{"fit": sorted(n for n in calibration if n not in held), "held": held}
                 for held in _stratified_groups(calibration, cond.__getitem__, n_inner, rng)]
        out.append({"test": test, "calibration": calibration, "inner": inner})
    return out
