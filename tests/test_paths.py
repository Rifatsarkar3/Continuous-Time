import pytest

from certalarm import paths


@pytest.mark.skipif(not paths.DUAL_STREAM.exists(), reason="source datasets not present")
def test_sources_exist():
    # every source the experiments read; the raw XJTU-SY files serve only the finished E1 audit (see paths.py)
    for p in (paths.PHM2012, paths.PRONOSTIA_PROCESSED, paths.XJTU_PROCESSED):
        assert p.exists(), p


def test_derived_dirs_are_inside_project():
    assert paths.DATA.parent == paths.ROOT
    assert paths.RESULTS.parent == paths.ROOT


def test_dual_stream_can_be_overridden(monkeypatch):
    import importlib

    monkeypatch.setenv("CERTALARM_DUAL_STREAM", str(paths.ROOT))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DUAL_STREAM == paths.ROOT
    finally:
        monkeypatch.delenv("CERTALARM_DUAL_STREAM")
        importlib.reload(paths)
