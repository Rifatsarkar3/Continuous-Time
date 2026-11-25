"""Write-once registry for outer-test evaluations.

The protocol says the test assets are touched once. Before a test evaluation, the exact
configuration and the trained weights are hashed and appended to a journal. A repeat evaluation
of the same hash and split is refused, and so is any evaluation of weights or a configuration
that was never registered - which is what a quiet retrain after a disappointing result would
look like.
"""
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import paths

REG = paths.RESULTS / "registry.jsonl"
STALE_LOCK_SECONDS = 600


def _exactly(o):
    """Serialise a config value losslessly, or refuse it.

    Never fall back to `str(o)`: numpy abbreviates a long array's repr with an ellipsis, so two
    arrays differing only in the hidden middle stringify identically and hash the same. That
    would silently defeat the one thing this module exists to detect.
    """
    if isinstance(o, np.ndarray):
        return {"dtype": str(o.dtype), "values": o.tolist()}
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} cannot be hashed exactly; registry configs must hold "
                    f"JSON-native values, numpy arrays or scalars")


def _hash_config(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_exactly).encode()).hexdigest()


def weights_hash(model):
    """Hash the raw bytes of every parameter, tagged by name, shape and dtype.

    The bytes are read through a uint8 view rather than `.numpy()`, which has no bfloat16 dtype
    to convert into and would raise instead of hashing.
    """
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(f"{tuple(t.shape)}{t.dtype}".encode())
        h.update(t.flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def _lock(path):
    lk = str(path) + ".lock"
    for _ in range(2000):
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            # A crash between acquiring and releasing would otherwise block every later run.
            try:
                if time.time() - os.path.getmtime(lk) > STALE_LOCK_SECONDS:
                    os.remove(lk)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
    else:
        raise TimeoutError(f"could not acquire {lk}")
    try:
        yield
    finally:
        os.close(fd)
        os.remove(lk)


def _entries(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _key(config, model):
    return {"config": _hash_config(config), "weights": weights_hash(model)}


def register(config, model, path=None):
    path = Path(path or REG)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _key(config, model)
    with _lock(path), path.open("a") as f:
        f.write(json.dumps({"event": "register", "time": time.time(), **key,
                            "tag": config.get("tag", "")}) + "\n")
    return key


def check(config, model, split, path=None):
    """Claim the one permitted evaluation of `split` for this config and these weights."""
    path = Path(path or REG)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _key(config, model)
    with _lock(path):
        entries = _entries(path)
        if not any(e["event"] == "register" and e["config"] == key["config"]
                   and e["weights"] == key["weights"] for e in entries):
            raise PermissionError("unregistered or modified config/weights - evaluation refused")
        if any(e["event"] == f"eval:{split}" and e["config"] == key["config"]
               and e["weights"] == key["weights"] for e in entries):
            raise PermissionError(f"split '{split}' already evaluated for this registration")
        with path.open("a") as f:
            f.write(json.dumps({"event": f"eval:{split}", "time": time.time(), **key}) + "\n")
    return True
