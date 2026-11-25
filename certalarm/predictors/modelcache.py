"""On-disk cache of trained predictors, so an interrupted run does not retrain what it finished.

Training dominates the cost of a run - minutes per model, hundreds of models - so this is the
checkpoint that actually saves time. The cache key carries the exact sorted list of fit-asset
names, which is what makes reuse safe: a cached model can only ever be returned for the same set
of bearings it was trained on, so it can never surface in a fold where it would have seen a
bearing it must not.

The key also carries every setting that changes the weights (seed, and whatever the caller passes
as `extra`: the observation process for a predictor that trains under it, the epoch budget, the
DANN flag). Anything that changes training must appear there, or a stale model will be reused.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

from .. import paths

DIR = paths.DATA / "model_cache"


def key(predictor, dataset, seed, fit_names, extra=None):
    payload = {"predictor": predictor, "dataset": dataset, "seed": int(seed),
               "fit": sorted(fit_names), "extra": extra or {}}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _save(state, path):
    """Write via a temporary file so an interrupted save cannot leave a half-written model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def cached(cache_key, make_empty, train_fn, enabled=True):
    """Return the cached model for `cache_key`, training and storing it on a miss.

    `make_empty` builds an untrained module of the right shape to load weights into; `train_fn`
    trains one from scratch. A cache file that fails to load is treated as a miss and retrained,
    so a corrupt file costs time but never correctness.
    """
    if not enabled:
        return train_fn()
    path = DIR / f"{cache_key}.pt"
    if path.exists():
        try:
            model = make_empty()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            return model
        except Exception:
            pass
    model = train_fn()
    _save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return model
