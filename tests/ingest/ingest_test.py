import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "ingest", Path(__file__).resolve().parents[2] / "src" / "ingest" / "ingest.py"
)
ingest_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_mod)

DAY = date(2026, 10, 30)


@pytest.fixture
def dirs(tmp_path):
    landing, raw = tmp_path / "landing", tmp_path / "raw" / "m5"
    landing.mkdir()
    (landing / "calendar.csv").write_bytes(b"date,wm_yr_wk\n2011-01-29,11101\n")
    (landing / "notes.txt").write_text("not a csv")
    return landing, raw


def test_copies_csvs_into_hive_partition_with_manifest(dirs):
    landing, raw = dirs
    manifest = ingest_mod.ingest(DAY, str(landing), str(raw))

    partition = raw / "ingest_date=2026-10-30"
    assert sorted(p.name for p in partition.iterdir()) == ["_manifest.json", "calendar.csv"]
    assert json.loads((partition / "_manifest.json").read_text()) == manifest
    data = (landing / "calendar.csv").read_bytes()
    assert manifest["files"] == [
        {"name": "calendar.csv", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    ]


def test_refuses_to_overwrite_partition_unless_forced(dirs):
    landing, raw = dirs
    ingest_mod.ingest(DAY, str(landing), str(raw))
    with pytest.raises(FileExistsError):
        ingest_mod.ingest(DAY, str(landing), str(raw))
    ingest_mod.ingest(DAY, str(landing), str(raw), force=True)


def test_fails_when_landing_has_no_csvs(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_mod.ingest(DAY, str(tmp_path), str(tmp_path / "raw"))
