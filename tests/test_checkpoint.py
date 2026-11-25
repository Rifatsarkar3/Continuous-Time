"""Tests for result checkpointing and the on-disk model cache."""
import json
import zlib

import numpy as np
import pytest
import torch

from certalarm import checkpoint
from certalarm.predictors import modelcache


def _unit(fold=0, seed=42):
    return {"dataset": "pronostia", "predictor": "ctgru", "process": "full",
            "seed": seed, "fold": fold}


def test_an_appended_unit_comes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    checkpoint.append("t", _unit(), [{"rule": "crc", "late": 0.1}])
    done = checkpoint.completed("t")
    assert list(done) == [checkpoint.unit_key(_unit())]
    assert done[checkpoint.unit_key(_unit())][0]["late"] == 0.1


def test_units_are_distinguished_by_every_field(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    checkpoint.append("t", _unit(fold=0), [{"a": 1}])
    checkpoint.append("t", _unit(fold=1), [{"a": 2}])
    checkpoint.append("t", _unit(fold=0, seed=7), [{"a": 3}])
    assert len(checkpoint.completed("t")) == 3


def test_unit_key_does_not_depend_on_dict_order():
    a = {"dataset": "p", "fold": 1, "seed": 42}
    b = {"seed": 42, "dataset": "p", "fold": 1}
    assert checkpoint.unit_key(a) == checkpoint.unit_key(b)


def test_an_unknown_tag_is_simply_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    assert checkpoint.completed("never-run") == {}


def test_a_truncated_final_line_does_not_lose_earlier_units(tmp_path, monkeypatch):
    # A crash mid-write leaves a partial line; everything before it must survive.
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    checkpoint.append("t", _unit(fold=0), [{"a": 1}])
    checkpoint.append("t", _unit(fold=1), [{"a": 2}])
    p = checkpoint.path("t")
    p.write_text(p.read_text() + '{"key": "half-writt')
    done = checkpoint.completed("t")
    assert len(done) == 2


def test_rows_survive_a_round_trip_through_json(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    rows = [{"late": np.float64(0.25), "theta": None, "certifiable": True}]
    checkpoint.append("t", _unit(), rows)
    got = checkpoint.completed("t")[checkpoint.unit_key(_unit())][0]
    assert got["late"] == 0.25 and got["theta"] is None and got["certifiable"] is True


def _tiny():
    torch.manual_seed(0)
    return torch.nn.Linear(3, 2)


def test_a_cached_model_is_not_retrained(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcache, "DIR", tmp_path)
    calls = []

    def train():
        calls.append(1)
        return _tiny()

    k = modelcache.key("ctgru", "pronostia", 42, ["B1", "B2"])
    first = modelcache.cached(k, lambda: torch.nn.Linear(3, 2), train)
    second = modelcache.cached(k, lambda: torch.nn.Linear(3, 2), train)
    assert len(calls) == 1
    assert torch.allclose(first.weight, second.weight)


def test_the_key_changes_with_the_fit_set():
    a = modelcache.key("ctgru", "pronostia", 42, ["B1", "B2"])
    b = modelcache.key("ctgru", "pronostia", 42, ["B1", "B3"])
    assert a != b


def test_the_key_ignores_the_order_of_the_fit_set():
    assert (modelcache.key("ctgru", "pronostia", 42, ["B2", "B1"])
            == modelcache.key("ctgru", "pronostia", 42, ["B1", "B2"]))


@pytest.mark.parametrize("field", ["predictor", "dataset", "seed", "extra"])
def test_every_training_relevant_field_changes_the_key(field):
    base = dict(predictor="ctgru", dataset="pronostia", seed=42, fit_names=["B1"],
                extra={"process": "full"})
    other = dict(base)
    other[field] = {"predictor": "dsaflite", "dataset": "xjtu", "seed": 43,
                    "extra": {"process": "mixture"}}[field]
    assert modelcache.key(**base) != modelcache.key(**other)


def test_a_corrupt_cache_file_is_retrained_rather_than_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcache, "DIR", tmp_path)
    k = modelcache.key("ctgru", "pronostia", 42, ["B1"])
    modelcache.cached(k, lambda: torch.nn.Linear(3, 2), _tiny)
    (tmp_path / f"{k}.pt").write_bytes(b"not a torch file")
    calls = []

    def train():
        calls.append(1)
        return _tiny()

    modelcache.cached(k, lambda: torch.nn.Linear(3, 2), train)
    assert calls == [1]


def test_caching_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcache, "DIR", tmp_path)
    calls = []

    def train():
        calls.append(1)
        return _tiny()

    k = modelcache.key("ctgru", "pronostia", 42, ["B1"])
    modelcache.cached(k, lambda: torch.nn.Linear(3, 2), train, enabled=False)
    modelcache.cached(k, lambda: torch.nn.Linear(3, 2), train, enabled=False)
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.pt"))


# --- end-to-end: crash partway through a run, then resume -------------------------------------

class _FakeAsset:
    def __init__(self, name, condition, n_min=30, per_min=6, seed=0):
        rng = np.random.default_rng(seed)
        self.name, self.condition, self.per_min = name, condition, per_min
        self.dataset = "pronostia"
        self.windows = rng.standard_normal((n_min * per_min, 32)).astype(np.float32)

    @property
    def n_min(self):
        return len(self.windows) // self.per_min

    @property
    def rul(self):
        return np.linspace(1.0, 0.0, self.n_min, dtype=np.float32)


class _FakePredictor:
    """Counts fits, and can be told to blow up after a given number of folds."""

    name = "fake"

    def __init__(self, crash_after=None):
        self.fits = 0
        self.crash_after = crash_after
        self.folds = 0

    def fit(self, fit_assets, seed, process):
        # Deterministic in the fit set, like the real predictors: the registry keys on weights,
        # so a randomly initialised fake would look like a different model on every run.
        self.fits += 1
        names = sorted(a.name for a in fit_assets)
        torch.manual_seed(zlib.crc32(f"{seed}|{'|'.join(names)}".encode()))
        return {"model": torch.nn.Linear(2, 1), "trained_on": names}

    def predict(self, model, asset, M):
        self.folds += 0
        return asset.rul.astype(float) + 0.02, np.full(asset.n_min, 0.05)


def _run_module(monkeypatch, tmp_path, fleet):
    import experiments.e2_main_alarms as e2
    monkeypatch.setattr(checkpoint, "DIR", tmp_path / "ckpt")
    monkeypatch.setattr(e2.registry, "REG", tmp_path / "registry.jsonl")
    monkeypatch.setattr(e2.assets, "load", lambda ds: fleet)
    return e2


def test_a_crashed_run_resumes_without_redoing_finished_units(tmp_path, monkeypatch):
    fleet = [_FakeAsset(f"Bearing{c}_{i}", c, seed=10 * c + i)
             for c in (1, 2, 3) for i in range(1, 4)]
    e2 = _run_module(monkeypatch, tmp_path, fleet)

    crashing = _FakePredictor()
    real_fit = crashing.fit
    state = {"n": 0}

    def exploding_fit(fit_assets, seed, process):
        state["n"] += 1
        if state["n"] > 8:                    # partway through the second fold
            raise RuntimeError("simulated crash")
        return real_fit(fit_assets, seed, process)

    crashing.fit = exploding_fit
    with pytest.raises(RuntimeError, match="simulated crash"):
        e2.run("pronostia", crashing, [42], ("full",), n_rep=1, ckpt_tag="t")

    finished = len(checkpoint.completed("t"))
    assert finished >= 1, "at least one unit should have been checkpointed before the crash"

    # Resume with a fresh predictor: the finished units must not be refitted.
    resumed = _FakePredictor()
    rows = e2.run("pronostia", resumed, [42], ("full",), n_rep=1, ckpt_tag="t")

    assert len(checkpoint.completed("t")) == 5          # all five outer folds now done
    assert resumed.fits == (5 - finished) * 4           # only the unfinished folds were refitted
    assert {r["fold"] for r in rows} == {0, 1, 2, 3, 4}


def test_resuming_does_not_spend_a_second_registry_evaluation(tmp_path, monkeypatch):
    # The registry permits one test evaluation per model. If resume re-ran a finished unit it
    # would be refused, so a clean resume proves the finished units were genuinely skipped.
    fleet = [_FakeAsset(f"Bearing{c}_{i}", c, seed=10 * c + i)
             for c in (1, 2, 3) for i in range(1, 4)]
    e2 = _run_module(monkeypatch, tmp_path, fleet)

    e2.run("pronostia", _FakePredictor(), [42], ("full",), n_rep=1, ckpt_tag="t")
    rows = e2.run("pronostia", _FakePredictor(), [42], ("full",), n_rep=1, ckpt_tag="t")
    assert {r["fold"] for r in rows} == {0, 1, 2, 3, 4}


def test_a_rerun_without_a_checkpoint_is_refused_by_the_registry(tmp_path, monkeypatch):
    # Without checkpointing, re-running the same folds must still be blocked - the resume path
    # is an exemption for work already done, not a hole in the write-once guarantee.
    fleet = [_FakeAsset(f"Bearing{c}_{i}", c, seed=10 * c + i)
             for c in (1, 2, 3) for i in range(1, 4)]
    e2 = _run_module(monkeypatch, tmp_path, fleet)
    p = _FakePredictor()
    e2.run("pronostia", p, [42], ("full",), n_rep=1, ckpt_tag=None)
    with pytest.raises(PermissionError):
        e2.run("pronostia", p, [42], ("full",), n_rep=1, ckpt_tag=None)


def test_a_dsaflite_model_reloaded_from_the_cache_can_still_predict(tmp_path, monkeypatch):
    """The resume path, end to end: save, reload in the state a fresh process sees, then PREDICT.

    Checking that reloaded weights are identical is not enough. The cache reloads onto the CPU
    while prediction targets the GPU when one exists, so a bit-identical model could still crash
    the first time it was used - which would have broken every retry of a crashed run.
    """
    from certalarm.predictors import dsaf_train
    from certalarm.predictors.dsaflite import DSAFLite

    monkeypatch.setattr(modelcache, "DIR", tmp_path)
    make = lambda: DSAFLite(feat_dim=dsaf_train.RECIPE["feat_dim"],
                            dropout=dsaf_train.RECIPE["dropout"])
    k = modelcache.key("dsaflite", "pronostia", 42, ["B1"])
    torch.manual_seed(0)
    modelcache.cached(k, make, make)                       # first call trains and saves
    reloaded = modelcache.cached(k, make, lambda: pytest.fail("should have been a cache hit"))
    assert next(reloaded.parameters()).device.type == "cpu"

    class _A:
        per_min = 6
        windows = np.random.default_rng(0).standard_normal((60, 2560)).astype(np.float32)
    mu, sigma = dsaf_train.predict_minutes(reloaded, _A(), mc_passes=2)
    assert mu.shape == (10,) and np.all(np.isfinite(mu)) and np.all(sigma >= 0)
