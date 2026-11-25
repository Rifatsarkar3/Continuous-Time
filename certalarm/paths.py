"""Filesystem locations. Source datasets are read-only; everything derived goes under DATA."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
_local = DATA / "source_dir.txt"          # git-ignored, one line: the directory that holds the source datasets
DUAL_STREAM = Path(os.environ.get("CERTALARM_DUAL_STREAM")
                   or (_local.read_text(encoding="utf-8").strip() if _local.exists() else DATA / "sources"))
PHM2012 = DUAL_STREAM / "data" / "PRONOSTIA_PHM2012"
PRONOSTIA_PROCESSED = DUAL_STREAM / "Processed_PRONOSTIA"
# Raw XJTU-SY files: only the one-time E1 audit reads them (results/e1_data_audit.json); re-running that audit
# needs the dataset downloaded again.
XJTU_RAW = DUAL_STREAM / "data" / "XJTU_SY_extracted" / "XJTU-SY_Bearing_Datasets"
XJTU_PROCESSED = DUAL_STREAM / "Processed_XJTU"
