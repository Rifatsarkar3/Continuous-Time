"""Tests for the write-once test-evaluation registry."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from certalarm import registry


def _model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(3, 1)


def test_registered_weights_may_be_evaluated_once(tmp_path):
    p = tmp_path / "reg.jsonl"
    cfg, m = {"tag": "t"}, _model()
    registry.register(cfg, m, path=p)
    assert registry.check(cfg, m, "test", path=p) is True


def test_a_second_evaluation_of_the_same_split_is_refused(tmp_path):
    p = tmp_path / "reg.jsonl"
    cfg, m = {"tag": "t"}, _model()
    registry.register(cfg, m, path=p)
    registry.check(cfg, m, "test", path=p)
    with pytest.raises(PermissionError, match="already evaluated"):
        registry.check(cfg, m, "test", path=p)


def test_a_different_split_is_still_allowed(tmp_path):
    p = tmp_path / "reg.jsonl"
    cfg, m = {"tag": "t"}, _model()
    registry.register(cfg, m, path=p)
    registry.check(cfg, m, "fold0", path=p)
    assert registry.check(cfg, m, "fold1", path=p) is True


def test_unregistered_weights_are_refused(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(PermissionError, match="unregistered"):
        registry.check({"tag": "t"}, _model(), "test", path=p)


def test_weights_modified_after_registration_are_refused(tmp_path):
    p = tmp_path / "reg.jsonl"
    cfg, m = {"tag": "t"}, _model()
    registry.register(cfg, m, path=p)
    with torch.no_grad():
        m.weight.add_(1.0)
    with pytest.raises(PermissionError, match="unregistered"):
        registry.check(cfg, m, "test", path=p)


def test_a_changed_config_is_refused(tmp_path):
    p = tmp_path / "reg.jsonl"
    m = _model()
    registry.register({"tag": "t", "alpha": 0.1}, m, path=p)
    with pytest.raises(PermissionError, match="unregistered"):
        registry.check({"tag": "t", "alpha": 0.2}, m, "test", path=p)


def test_weights_hash_is_stable_and_distinguishes_models():
    a, b = _model(0), _model(0)
    assert registry.weights_hash(a) == registry.weights_hash(b)
    assert registry.weights_hash(a) != registry.weights_hash(_model(1))


def test_configs_differing_only_inside_a_long_array_do_not_collide():
    # numpy abbreviates a long array's repr, so a str() fallback would hash these identically.
    a = np.arange(2000)
    b = a.copy()
    b[1000] = 999999
    assert registry._hash_config({"g": a}) != registry._hash_config({"g": b})


def test_a_value_that_cannot_be_hashed_exactly_is_refused():
    with pytest.raises(TypeError, match="cannot be hashed exactly"):
        registry._hash_config({"f": object()})


def test_weights_hash_handles_bfloat16():
    m = _model()
    b = m.to(torch.bfloat16)
    assert len(registry.weights_hash(b)) == 64


def test_weights_hash_separates_models_that_differ_only_in_dtype():
    torch.manual_seed(0)
    a = torch.nn.Linear(3, 1)
    torch.manual_seed(0)
    b = torch.nn.Linear(3, 1).to(torch.float64)
    assert registry.weights_hash(a) != registry.weights_hash(b)


def test_a_stale_lock_is_reclaimed(tmp_path, monkeypatch):
    p = tmp_path / "reg.jsonl"
    lk = Path(str(p) + ".lock")
    lk.write_text("")
    os.utime(lk, (0, 0))   # far older than STALE_LOCK_SECONDS
    cfg, m = {"tag": "t"}, _model()
    registry.register(cfg, m, path=p)
    assert registry.check(cfg, m, "test", path=p) is True
