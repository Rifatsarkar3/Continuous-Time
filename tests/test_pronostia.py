import numpy as np
import pytest

from certalarm.data import pronostia as P


def write_acc(path, horiz, sep=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sep.join(["9", "39", "39", str(65664 + i), f"{h:.3f}", "0.100"]) for i, h in enumerate(horiz)]
    path.write_text("\n".join(rows) + "\n")


def test_read_acc_comma(tmp_path):
    f = tmp_path / "acc_00001.csv"
    write_acc(f, [0.5, -1.25, 2.0])
    assert np.allclose(P.read_acc(f), [0.5, -1.25, 2.0])
    assert P.read_acc(f).dtype == np.float32


def test_read_acc_semicolon(tmp_path):
    f = tmp_path / "acc_00001.csv"
    write_acc(f, [0.1, 0.2], sep=";")
    assert np.allclose(P.read_acc(f), [0.1, 0.2])


def test_acc_files_filters_temp_and_sorts(tmp_path):
    d = tmp_path / "Bearing1_4"
    for name in ("acc_00002.csv", "temp_00001.csv", "acc_00001.csv", "acc_00010.csv"):
        write_acc(d / name, [0.0])
    assert [f.name for f in P.acc_files(d)] == ["acc_00001.csv", "acc_00002.csv", "acc_00010.csv"]


def test_catalogue_has_17_bearings_in_3_conditions():
    expected = {
        "Bearing1_1": ("Learning_set", 1), "Bearing1_2": ("Learning_set", 1),
        "Bearing2_1": ("Learning_set", 2), "Bearing2_2": ("Learning_set", 2),
        "Bearing3_1": ("Learning_set", 3), "Bearing3_2": ("Learning_set", 3),
        "Bearing1_3": ("Full_Test_Set", 1), "Bearing1_4": ("Full_Test_Set", 1),
        "Bearing1_5": ("Full_Test_Set", 1), "Bearing1_6": ("Full_Test_Set", 1),
        "Bearing1_7": ("Full_Test_Set", 1), "Bearing2_3": ("Full_Test_Set", 2),
        "Bearing2_4": ("Full_Test_Set", 2), "Bearing2_5": ("Full_Test_Set", 2),
        "Bearing2_6": ("Full_Test_Set", 2), "Bearing2_7": ("Full_Test_Set", 2),
        "Bearing3_3": ("Full_Test_Set", 3),
    }
    assert P.BEARINGS == expected
    assert P.CONDITIONS == {1: (1800.0, 4000.0), 2: (1650.0, 4200.0), 3: (1500.0, 5000.0)}


def fake_root(tmp_path, n_learning=3, n_test=2, L=8):
    rng = np.random.default_rng(0)
    for k in range(n_learning):
        write_acc(tmp_path / "raw" / "Learning_set" / "Bearing1_1" / f"acc_{k + 1:05d}.csv", rng.normal(size=L))
    for k in range(n_test):
        write_acc(tmp_path / "raw" / "Full_Test_Set" / "Bearing1_4" / f"acc_{k + 1:05d}.csv",
                  rng.normal(size=L), sep=";")
    return tmp_path / "raw"


def test_build_bearing_labels_and_shape(tmp_path):
    root = fake_root(tmp_path)
    b = P.build_bearing(root, "Bearing1_1")
    assert b["windows"].shape == (3, 8)
    assert np.allclose(b["rul"], [1.0, 0.5, 0.0])
    assert b["condition"] == 1 and b["n"] == 3


def test_build_all_writes_npz_and_manifest(tmp_path):
    root = fake_root(tmp_path)
    out = tmp_path / "out"
    man = P.build_all(root, out, names=["Bearing1_1", "Bearing1_4"])
    assert man == {"Bearing1_1": {"n": 3, "condition": 1, "source": "Learning_set"},
                   "Bearing1_4": {"n": 2, "condition": 1, "source": "Full_Test_Set"}}
    z = np.load(out / "Bearing1_4.npz")
    assert z["windows"].shape == (2, 8) and np.allclose(z["rul"], [1.0, 0.0])
    assert (out / "manifest.json").exists()


def test_build_outputs_are_float32(tmp_path):
    root = fake_root(tmp_path)
    b = P.build_bearing(root, "Bearing1_1")
    assert b["windows"].dtype == np.float32 and b["rul"].dtype == np.float32
    P.build_all(root, tmp_path / "out", names=["Bearing1_1"])
    z = np.load(tmp_path / "out" / "Bearing1_1.npz")
    assert z["windows"].dtype == np.float32 and z["rul"].dtype == np.float32


def test_build_bearing_missing_files_raises(tmp_path):
    (tmp_path / "raw" / "Learning_set" / "Bearing1_1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        P.build_bearing(tmp_path / "raw", "Bearing1_1")


def test_build_all_empty_names_builds_nothing(tmp_path):
    root = fake_root(tmp_path)
    assert P.build_all(root, tmp_path / "out", names=[]) == {}


def test_audit_detects_extra_windows(tmp_path):
    root = fake_root(tmp_path)
    proc = tmp_path / "proc"
    good = P.build_bearing(root, "Bearing1_1")["windows"]
    bad = P.build_bearing(root, "Bearing1_4")["windows"]
    (proc / "Bearing1_1").mkdir(parents=True)
    (proc / "Bearing1_4").mkdir(parents=True)
    np.save(proc / "Bearing1_1" / "windows_vib.npy", good)
    np.save(proc / "Bearing1_4" / "windows_vib.npy", np.vstack([bad, 40 * np.ones((1, 8), np.float32)]))
    rep = P.audit_processed(root, proc, names=["Bearing1_1", "Bearing1_4"])
    assert rep["Bearing1_1"]["verdict"] == "ok"
    assert rep["Bearing1_4"]["verdict"] == "extra_windows_after_raw_end"
    assert rep["Bearing1_4"]["extra_windows"] == 1
    assert rep["Bearing1_4"]["raw_end_found_at_index"] == 1
    assert rep["Bearing1_4"]["rms_extra_median"] == 40.0
    assert rep["Bearing1_4"]["rms_extra_first"] == 40.0
    assert rep["Bearing1_4"]["rms_extra_last"] == 40.0


def test_audit_mismatch_cases(tmp_path):
    root = fake_root(tmp_path)
    w = P.build_bearing(root, "Bearing1_1")["windows"]
    cases = {"first": w.copy(), "interior": w.copy(), "short": w[:2].copy()}
    cases["first"][0] += 5.0
    cases["interior"][1] += 5.0
    for tag, arr in cases.items():
        d = tmp_path / "proc" / tag / "Bearing1_1"
        d.mkdir(parents=True)
        np.save(d / "windows_vib.npy", arr)
        rep = P.audit_processed(root, tmp_path / "proc" / tag, names=["Bearing1_1"])
        assert rep["Bearing1_1"]["verdict"] == "mismatch", tag


def test_audit_missing_raw_does_not_crash(tmp_path):
    root = tmp_path / "raw"
    (root / "Learning_set" / "Bearing1_1").mkdir(parents=True)
    d = tmp_path / "proc" / "Bearing1_1"
    d.mkdir(parents=True)
    np.save(d / "windows_vib.npy", np.zeros((2, 8), np.float32))
    rep = P.audit_processed(root, tmp_path / "proc", names=["Bearing1_1"])
    assert rep["Bearing1_1"]["verdict"] == "missing_raw"
