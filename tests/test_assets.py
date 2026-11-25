"""Tests for asset loading."""
import json

import numpy as np
import pytest

from certalarm import assets


def _write_pronostia(d, name, n_win, condition):
    d.mkdir(parents=True, exist_ok=True)
    w = np.arange(n_win * 4, dtype=np.float32).reshape(n_win, 4)
    np.savez(d / f"{name}.npz", windows=w, rul=np.linspace(1, 0, n_win, dtype=np.float32),
             condition=condition)
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    man[name] = {"n": n_win, "condition": condition, "source": "Learning_set"}
    (d / "manifest.json").write_text(json.dumps(man))


def test_minute_grid_drops_the_partial_final_minute(tmp_path):
    _write_pronostia(tmp_path, "Bearing1_1", 20, 1)   # 20 windows = 3 whole minutes + 2
    a, = assets.load_pronostia(data_dir=tmp_path)
    assert a.per_min == 6
    assert a.n_min == 3
    assert len(a.windows) == 18


def test_rul_runs_from_one_to_zero_on_the_minute_grid(tmp_path):
    _write_pronostia(tmp_path, "Bearing1_1", 24, 1)
    a, = assets.load_pronostia(data_dir=tmp_path)
    assert a.n_min == 4
    assert a.rul[0] == pytest.approx(1.0)
    assert a.rul[-1] == pytest.approx(0.0)
    assert np.all(np.diff(a.rul) < 0)


def test_condition_carries_speed_and_load(tmp_path):
    _write_pronostia(tmp_path, "Bearing2_1", 12, 2)
    a, = assets.load_pronostia(data_dir=tmp_path)
    assert a.condition == 2
    assert a.speed_load == (1650.0, 4200.0)


def test_named_subset_loads_only_those_assets(tmp_path):
    _write_pronostia(tmp_path, "Bearing1_1", 12, 1)
    _write_pronostia(tmp_path, "Bearing1_2", 12, 1)
    got = assets.load_pronostia(names=["Bearing1_2"], data_dir=tmp_path)
    assert [a.name for a in got] == ["Bearing1_2"]


def test_empty_name_list_loads_nothing(tmp_path):
    _write_pronostia(tmp_path, "Bearing1_1", 12, 1)
    assert assets.load_pronostia(names=[], data_dir=tmp_path) == []


def test_asset_shorter_than_one_minute_is_rejected(tmp_path):
    _write_pronostia(tmp_path, "Bearing1_1", 4, 1)   # fewer than 6 windows
    with pytest.raises(ValueError, match="at least one whole minute"):
        assets.load_pronostia(data_dir=tmp_path)


def test_xjtu_condition_comes_from_the_bearing_name(tmp_path):
    w = np.zeros((24, 4), dtype=np.float32)
    np.savez(tmp_path / "train_01_Bearing3_2.npz", vibration=w)
    a, = assets.load_xjtu(proc_dir=tmp_path)
    assert a.name == "Bearing3_2"
    assert a.condition == 3
    assert a.per_min == 12
    assert a.n_min == 2
