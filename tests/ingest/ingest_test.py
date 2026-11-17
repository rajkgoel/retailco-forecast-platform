import hashlib
import sys
import json
from datetime import date, timedelta

import pytest

from src.ingest import ingest as ingest_mod

DAY = date(2026, 10, 30)
NEXT_DAY = DAY + timedelta(days=1)


@pytest.fixture
def dirs(tmp_path):
    landing, raw = tmp_path / "landing", tmp_path / "raw" / "m5"
    landing.mkdir()
    (landing / "calendar.csv").write_bytes(b"date,wm_yr_wk\n2011-01-29,11101\n")
    (landing / "sell_prices.csv").write_bytes(b"store_id,item_id,wm_yr_wk,sell_price\nCA_1,X,11325,9.58\n")
    (landing / "notes.txt").write_text("not a csv")
    return landing, raw


def partition(raw, day=DAY):
    return raw / f"ingest_date={day.isoformat()}"


@pytest.fixture
def crash_on_second_copy(monkeypatch):
    """Return a function that makes the next run crash after copying its first file."""
    real = ingest_mod._copy_with_hash

    def arm():
        calls = []

        def flaky(*args, **kwargs):
            if calls:
                raise OSError("simulated crash")
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(ingest_mod, "_copy_with_hash", flaky)

    return arm


def test_copies_csvs_into_hive_partition_with_manifest(dirs):
    landing, raw = dirs
    status, manifest = ingest_mod.ingest(DAY, str(landing), str(raw))

    assert status == "ingested"
    assert sorted(p.name for p in partition(raw).iterdir()) == [
        "_manifest.json", "calendar.csv", "sell_prices.csv"
    ]
    assert json.loads((partition(raw) / "_manifest.json").read_text()) == manifest
    data = (landing / "calendar.csv").read_bytes()
    assert manifest["files"][0] == {
        "name": "calendar.csv", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
    }


def test_refuses_to_overwrite_complete_partition_unless_forced(dirs):
    landing, raw = dirs
    ingest_mod.ingest(DAY, str(landing), str(raw))
    with pytest.raises(FileExistsError):
        ingest_mod.ingest(DAY, str(landing), str(raw))
    assert (partition(raw) / "_manifest.json").exists()

    status, _ = ingest_mod.ingest(DAY, str(landing), str(raw), force=True)
    assert status == "ingested"
    assert (partition(raw) / "_manifest.json").exists()


def test_crash_during_forced_rerun_leaves_no_manifest(dirs, crash_on_second_copy):
    landing, raw = dirs
    ingest_mod.ingest(DAY, str(landing), str(raw))

    crash_on_second_copy()
    with pytest.raises(OSError, match="simulated crash"):
        ingest_mod.ingest(DAY, str(landing), str(raw), force=True)

    # The old manifest must be gone, so readers treat the partition as incomplete.
    assert not (partition(raw) / "_manifest.json").exists()


def test_forced_rerun_removes_files_from_earlier_delivery(dirs):
    landing, raw = dirs
    (landing / "promo.csv").write_text("item_id\nX\n")
    ingest_mod.ingest(DAY, str(landing), str(raw))
    (landing / "promo.csv").unlink()

    ingest_mod.ingest(DAY, str(landing), str(raw), force=True)
    assert not (partition(raw) / "promo.csv").exists()


def test_crashed_partition_is_rebuilt_without_force(dirs, crash_on_second_copy, monkeypatch):
    landing, raw = dirs
    crash_on_second_copy()
    with pytest.raises(OSError, match="simulated crash"):
        ingest_mod.ingest(DAY, str(landing), str(raw))
    assert not (partition(raw) / "_manifest.json").exists()

    monkeypatch.undo()  # restore the real copy function
    status, _ = ingest_mod.ingest(DAY, str(landing), str(raw))
    assert status == "ingested"
    assert (partition(raw) / "_manifest.json").exists()


def test_identical_delivery_on_later_date_is_skipped(dirs):
    landing, raw = dirs
    _, first = ingest_mod.ingest(DAY, str(landing), str(raw))

    status, manifest = ingest_mod.ingest(NEXT_DAY, str(landing), str(raw))
    assert status == "skipped"
    assert manifest == first
    assert not partition(raw, NEXT_DAY).exists()


def test_changed_delivery_on_later_date_is_ingested(dirs):
    landing, raw = dirs
    ingest_mod.ingest(DAY, str(landing), str(raw))
    (landing / "calendar.csv").write_bytes(b"date,wm_yr_wk\n2011-01-30,11101\n")  # same size

    status, _ = ingest_mod.ingest(NEXT_DAY, str(landing), str(raw))
    assert status == "ingested"
    assert (partition(raw, NEXT_DAY) / "_manifest.json").exists()


def test_incomplete_partition_is_not_used_to_skip(dirs):
    landing, raw = dirs
    partition(raw).mkdir(parents=True)
    (partition(raw) / "calendar.csv").write_bytes((landing / "calendar.csv").read_bytes())

    status, _ = ingest_mod.ingest(NEXT_DAY, str(landing), str(raw))
    assert status == "ingested"


def test_fails_when_landing_has_no_csvs(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_mod.ingest(DAY, str(tmp_path), str(tmp_path / "raw"))

def test_main_exits_2_when_partition_exists(dirs, monkeypatch, caplog):
    landing, raw = dirs
    ingest_mod.ingest(DAY, str(landing), str(raw))
    monkeypatch.setattr(sys, "argv", ["ingest", "--date", DAY.isoformat(),
                                      "--landing", str(landing), "--raw", str(raw)])
    with pytest.raises(SystemExit) as exc:
        ingest_mod.main()
    assert exc.value.code == 2
    assert "pass --force" in caplog.text
