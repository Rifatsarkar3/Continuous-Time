import numpy as np

from certalarm.data import xjtu as X

PER, WIN = 2, 4


def fake(tmp_path, n_files=11, lexicographic=False):
    rng = np.random.default_rng(0)
    d = tmp_path / "raw" / "35Hz12kN" / "Bearing1_1"
    d.mkdir(parents=True)
    data = {}
    for k in range(1, n_files + 1):
        h = rng.normal(size=PER * WIN).round(4)
        data[k] = h
        rows = ["Horizontal_vibration_signals,Vertical_vibration_signals"] + [f"{v},0.0" for v in h]
        (d / f"{k}.csv").write_text("\n".join(rows) + "\n")
    order = sorted(data, key=str) if lexicographic else sorted(data)
    windows = np.concatenate([data[k].reshape(PER, WIN) for k in order]).astype(np.float32)
    proc = tmp_path / "proc"
    proc.mkdir()
    np.savez(proc / "train_01_Bearing1_1.npz", vibration=windows, rul=np.zeros(len(windows)))
    return proc, tmp_path / "raw"


def test_numeric_order_ok(tmp_path):
    proc, raw = fake(tmp_path)
    rep = X.audit_processed(proc, raw, per_file=PER, win=WIN)
    assert rep["Bearing1_1"]["verdict"] == "ok" and rep["Bearing1_1"]["files"] == 11


def test_lexicographic_order_detected(tmp_path):
    proc, raw = fake(tmp_path, lexicographic=True)
    rep = X.audit_processed(proc, raw, per_file=PER, win=WIN)
    assert rep["Bearing1_1"]["verdict"] == "mismatch"
    assert rep["Bearing1_1"]["second_file"] is False


def test_missing_raw_reported(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    np.savez(proc / "train_01_Bearing1_1.npz", vibration=np.zeros((4, WIN), np.float32), rul=np.zeros(4))
    (tmp_path / "raw" / "35Hz12kN" / "Bearing1_1").mkdir(parents=True)
    rep = X.audit_processed(proc, tmp_path / "raw", per_file=PER, win=WIN)
    assert rep["Bearing1_1"]["verdict"] == "missing_raw"


def test_count_mismatch_alone_fails(tmp_path):
    proc, raw = fake(tmp_path)
    z = np.load(proc / "train_01_Bearing1_1.npz")
    np.savez(proc / "train_01_Bearing1_1.npz", vibration=z["vibration"][:-1], rul=z["rul"][:-1])
    rep = X.audit_processed(proc, raw, per_file=PER, win=WIN)
    assert rep["Bearing1_1"]["count_ok"] is False and rep["Bearing1_1"]["verdict"] == "mismatch"


def test_within_file_window_error_detected(tmp_path):
    proc, raw = fake(tmp_path)
    z = np.load(proc / "train_01_Bearing1_1.npz")
    v = z["vibration"].copy()
    v[1] += 1.0
    np.savez(proc / "train_01_Bearing1_1.npz", vibration=v, rul=z["rul"])
    rep = X.audit_processed(proc, raw, per_file=PER, win=WIN)
    assert rep["Bearing1_1"]["first"] is False and rep["Bearing1_1"]["verdict"] == "mismatch"


def test_short_raw_file_is_mismatch_not_crash(tmp_path):
    proc, raw = fake(tmp_path)
    f = raw / "35Hz12kN" / "Bearing1_1" / "11.csv"
    f.write_text("Horizontal_vibration_signals,Vertical_vibration_signals\n0.1,0.0\n")
    rep = X.audit_processed(proc, raw, per_file=PER, win=WIN)
    assert rep["Bearing1_1"]["last_file"] is False and rep["Bearing1_1"]["verdict"] == "mismatch"
