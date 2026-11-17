import hashlib
import io
import sys
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from src.ingest import ingest as ingest_mod
from src.validate import validate as v

DAY = date(2026, 10, 30)
NEXT_DAY = DAY + timedelta(days=1)

# A small contract used by the unit tests, so they don't break when the real one changes.
SALES_SPEC = {
    "key": ["id"],
    "columns": {"id": {"type": "string"}, "store_id": {"type": "string"}},
    "day_columns": {"pattern": r"^d_(\d+)$", "contiguous_from": 1, "type": "int", "min": 0,
                    "not_fewer_than_previous": True},
    "volume": {"max_row_drop_vs_previous": 0.05},
}


def sales(**day_values) -> pd.DataFrame:
    df = pd.DataFrame({"id": pd.array(["a", "b"], dtype="string"),
                       "store_id": pd.array(["CA_1", "TX_1"], dtype="string")})
    for col, vals in day_values.items():
        df[col] = vals
    return df


# =========================================================================== unit: structure

class TestRequiredColumns:
    def test_all_present(self):
        assert v.check_required_columns("s", sales(d_1=[1, 2]), SALES_SPEC).ok

    def test_missing_column_is_critical_and_named(self):
        r = v.check_required_columns("s", sales().drop(columns="store_id"), SALES_SPEC)
        assert (r.failed, r.total, r.severity) == (1, 2, "critical")
        assert "store_id" in r.detail

    def test_extra_column_is_fine(self):
        assert v.check_required_columns("s", sales().assign(extra=1), SALES_SPEC).ok

    def test_names_are_not_normalised(self):
        df = sales().rename(columns={"store_id": "Store_id"})
        assert not v.check_required_columns("s", df, SALES_SPEC).ok


class TestDayColumns:
    def test_contiguous_columns_pass_and_come_back_in_order(self):
        r, cols = v.check_day_columns("s", sales(d_1=[1, 1], d_2=[1, 1], d_3=[1, 1]), SALES_SPEC)
        assert r.ok and cols == ["d_1", "d_2", "d_3"]

    def test_gap_is_reported(self):
        r, _ = v.check_day_columns("s", sales(d_1=[1, 1], d_2=[1, 1], d_4=[1, 1]), SALES_SPEC)
        assert not r.ok and "[3]" in r.detail

    def test_wrong_start(self):
        r, _ = v.check_day_columns("s", sales(d_2=[1, 1], d_3=[1, 1]), SALES_SPEC)
        assert not r.ok and "starts at 2" in r.detail

    def test_out_of_order(self):
        r, cols = v.check_day_columns("s", sales(d_1=[1, 1], d_3=[1, 1], d_2=[1, 1]), SALES_SPEC)
        assert not r.ok and "order" in r.detail
        assert cols == ["d_1", "d_2", "d_3"]

    def test_no_day_columns(self):
        r, cols = v.check_day_columns("s", sales(), SALES_SPEC)
        assert not r.ok and cols == []

    def test_repeated_header_renamed_by_pandas_is_caught(self):
        df = pd.read_csv(io.StringIO("id,store_id,d_1,d_2,d_2\na,CA_1,1,2,3\n"))
        assert "d_2.1" in df.columns  # pandas silently renames the repeat
        r, _ = v.check_day_columns("s", df, SALES_SPEC)
        assert not r.ok and "repeated" in r.detail


class TestTypes:
    SPEC = {"columns": {"n": {"type": "int"}, "price": {"type": "float"},
                        "event": {"type": "string", "nullable": True}},
            "day_columns": {"type": "int"}}

    def results(self, df, day_cols=()):
        return {r.check: r for r in v.check_types("f", df, self.SPEC, list(day_cols))}

    def test_valid_types(self):
        df = pd.DataFrame({"n": [1], "price": [2.5], "event": pd.array(["x"], dtype="string")})
        assert all(r.ok for r in self.results(df).values())

    def test_text_in_int_column_fails(self):
        df = pd.DataFrame({"n": ["abc"], "price": [1.0], "event": [None]})
        assert not self.results(df)["type:n"].ok

    def test_float_column_of_whole_numbers_passes(self):
        df = pd.DataFrame({"n": [1], "price": [3], "event": [None]})
        assert self.results(df)["type:price"].ok

    def test_all_empty_nullable_column_passes(self):
        df = pd.read_csv(io.StringIO("n,price,event\n1,1.0,\n"))
        assert df["event"].dtype == "float64"
        assert self.results(df)["type:event"].ok

    def test_text_in_day_column_fails(self):
        df = pd.DataFrame({"n": [1], "price": [1.0], "event": [None], "d_1": [1], "d_2": ["x"]})
        r = self.results(df, ["d_1", "d_2"])["type:day_columns"]
        assert r.failed == 1 and "d_2" in r.detail

    def test_unknown_type_in_contract_raises(self):
        with pytest.raises(ValueError, match="unknown type"):
            v.check_types("f", pd.DataFrame({"n": [1]}), {"columns": {"n": {"type": "integer"}}}, [])


# =========================================================================== unit: keys and volume

class TestKey:
    def test_unique(self):
        assert v.check_key("s", sales(), ["id"]).ok

    def test_duplicate_counts_every_copy(self):
        df = pd.DataFrame({"id": ["a", "a", "b"]})
        r = v.check_key("s", df, ["id"])
        assert (r.failed, r.severity) == (2, "critical")

    def test_composite_key(self):
        df = pd.DataFrame({"store_id": ["CA_1", "CA_1"], "item_id": ["X", "X"], "wm_yr_wk": [1, 2]})
        key = ["store_id", "item_id", "wm_yr_wk"]
        assert v.check_key("p", df, key).ok
        assert not v.check_key("p", df.assign(wm_yr_wk=[1, 1]), key).ok

    def test_null_in_key_fails(self):
        assert v.check_key("s", pd.DataFrame({"id": ["a", None]}), ["id"]).failed == 1


class TestSequence:
    def test_contiguous_in_any_row_order(self):
        df = pd.DataFrame({"d": ["d_2", "d_1", "d_3"]})
        assert v.check_sequence("c", df, "d", r"^d_(\d+)$", 1).ok

    def test_gap(self):
        assert not v.check_sequence("c", pd.DataFrame({"d": ["d_1", "d_3"]}), "d", r"^d_(\d+)$", 1).ok

    def test_value_not_matching_pattern(self):
        r = v.check_sequence("c", pd.DataFrame({"d": ["d_1", "day2"]}), "d", r"^d_(\d+)$", 1)
        assert not r.ok and "day2" in r.detail


class TestDateContiguous:
    @pytest.mark.parametrize("dates, failed", [
        (["2011-01-29", "2011-01-30", "2011-01-31"], 0),
        (["2011-01-29", "2011-01-31"], 1),                 # gap
        (["2011-01-29", "2011-01-29", "2011-01-30"], 1),   # repeat
        (["2011-01-29", "2011-13-01"], 1),                 # unparsable
    ])
    def test_dates(self, dates, failed):
        df = pd.DataFrame({"date": pd.array(dates, dtype="string")})
        assert v.check_date_contiguous("c", df, "date", "%Y-%m-%d").failed == failed


class TestVolume:
    def failed(self, current, previous):
        return {r.check: r.failed for r in v.check_volume("s", current, previous, SALES_SPEC)}

    def test_first_delivery_has_nothing_to_compare(self):
        assert self.failed({"rows": 1, "day_columns": 1}, None) == {"volume": 0}

    def test_small_drop_passes(self):
        assert self.failed({"rows": 96, "day_columns": 5}, {"rows": 100, "day_columns": 5})["volume:row_drop"] == 0

    def test_large_drop_fails(self):
        assert self.failed({"rows": 94, "day_columns": 5}, {"rows": 100, "day_columns": 5})["volume:row_drop"] == 1

    def test_fewer_day_columns_fails(self):
        assert self.failed({"rows": 100, "day_columns": 4}, {"rows": 100, "day_columns": 5})["volume:day_columns"] == 1

    def test_not_fewer_than_previous(self):
        spec = {"volume": {"not_fewer_than_previous": True}}
        [r] = v.check_volume("p", {"rows": 99}, {"rows": 100}, spec)
        assert r.failed == 1


# =========================================================================== unit: values

class TestValues:
    def test_nulls(self):
        df = pd.DataFrame({"x": ["a", None]})
        assert v.check_nulls("f", df, "x", nullable=False)[0].failed == 1
        r, mask = v.check_nulls("f", df, "x", nullable=True)
        assert r.ok and not mask.any()

    def test_allowed_ignores_nulls(self):
        df = pd.DataFrame({"s": ["CA_1", "XX_9", None]})
        r, mask = v.check_allowed("f", df, "s", ["CA_1"])
        assert r.failed == 1 and mask.tolist() == [False, True, False]

    def test_range(self):
        df = pd.DataFrame({"p": [0.0, 0.01, 5.0, None]})
        r, mask = v.check_range("f", df, "p", min_=0.01)
        assert mask.tolist() == [True, False, False, False]
        r, mask = v.check_range("f", df, "p", max_=1.0)
        assert mask.tolist() == [False, False, True, False]

    def test_pattern_ignores_nulls(self):
        df = pd.DataFrame({"id": pd.array(["x_evaluation", "x_validation", None], dtype="string")})
        r, mask = v.check_pattern("f", df, "id", r"^.+_evaluation$")
        assert mask.tolist() == [False, True, False]

    def test_null_together(self):
        df = pd.DataFrame({"name": ["SuperBowl", None, "Easter", None],
                           "type": ["Sporting", None, None, "Cultural"]})
        _, mask = v.check_null_together("c", df, "name", "type")
        assert mask.tolist() == [False, False, True, True]

    def test_day_values_returns_bad_cells(self):
        df = sales(d_1=[1, 2], d_2=[0, -1], d_3=[4, 5])
        r, cells = v.check_day_values("s", df, ["d_1", "d_2", "d_3"], SALES_SPEC)
        assert (r.failed, r.total) == (1, 6)
        assert cells.to_dict("records") == [{"id": "b", "d": "d_2", "value": -1}]


class TestPolicy:
    def test_escalates_above_limit(self):
        r = v.escalate(v.CheckResult("c", "f", "row", 2, 100), 0.005)
        assert r.severity == "critical" and "escalated" in r.detail

    def test_stays_row_level_at_or_below_limit(self):
        assert v.escalate(v.CheckResult("c", "f", "row", 1, 200), 0.005).severity == "row"

    def test_critical_file_overrides_row_severity(self):
        r = v.apply_policy(v.CheckResult("c", "calendar.csv", "row", 1, 10_000), {"severity": "critical"}, 0.5)
        assert r.severity == "critical"

    def test_run_stage_records_all_then_stops_on_critical(self):
        report = {"stages": []}
        results = [v.CheckResult("a", "f", "critical", 1, 1), v.CheckResult("b", "f", "critical", 1, 1)]
        with pytest.raises(v.StageFailed):
            v.run_stage("s", results, report)
        assert len(report["stages"][0]["checks"]) == 2

    def test_row_failures_do_not_stop_a_stage(self):
        v.run_stage("s", [v.CheckResult("a", "f", "row", 1, 1)], {"stages": []})


class TestRelationship:
    FRAMES = {
        "sales": pd.DataFrame({"store_id": ["CA_1", "TX_1"], "item_id": ["X", "Y"]}),
        "prices": pd.DataFrame({"store_id": ["CA_1", "CA_1", "TX_1"], "item_id": ["X", "Z", "Y"],
                                "wm_yr_wk": [1, 1, 9]}),
        "calendar": pd.DataFrame({"d": ["d_1", "d_2"], "wm_yr_wk": [1, 1]}),
    }

    def rel(self, src, src_cols, dst, dst_cols, **kw):
        return {"name": "r", "from": {"file": src, "columns": src_cols},
                "to": {"file": dst, "columns": dst_cols}, **kw}

    def test_composite_key_orphan(self):
        r, mask = v.check_relationship(
            self.rel("prices", ["store_id", "item_id"], "sales", ["store_id", "item_id"]), self.FRAMES, [])
        assert r.severity == "row" and mask.tolist() == [False, True, False]

    def test_single_column(self):
        r, mask = v.check_relationship(self.rel("prices", ["wm_yr_wk"], "calendar", ["wm_yr_wk"]), self.FRAMES, [])
        assert mask.tolist() == [False, False, True]

    def test_duplicate_keys_on_right_do_not_misalign_the_mask(self):
        # calendar has wm_yr_wk=1 twice; without drop_duplicates the merge would return 5 rows
        _, mask = v.check_relationship(self.rel("prices", ["wm_yr_wk"], "calendar", ["wm_yr_wk"]), self.FRAMES, [])
        assert len(mask) == 3

    def test_day_columns_missing_from_calendar(self):
        r, mask = v.check_relationship(
            self.rel("sales", "day_columns", "calendar", ["d"], severity="critical"),
            self.FRAMES, ["d_1", "d_2", "d_3"])
        assert mask is None and r.failed == 1 and "d_3" in r.detail


# =========================================================================== end to end

CALENDAR = """date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI
2011-01-29,11101,Saturday,1,1,2011,d_1,,,,,0,0,0
2011-01-30,11101,Sunday,2,1,2011,d_2,SuperBowl,Sporting,,,0,0,0
2011-01-31,11101,Monday,3,1,2011,d_3,,,,,1,1,0
"""
SALES = """id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2,d_3
FOODS_1_001_CA_1_evaluation,FOODS_1_001,FOODS_1,FOODS,CA_1,CA,1,0,2
HOBBIES_1_001_TX_1_evaluation,HOBBIES_1_001,HOBBIES_1,HOBBIES,TX_1,TX,0,3,1
"""
PRICES = """store_id,item_id,wm_yr_wk,sell_price
CA_1,FOODS_1_001,11101,2.5
TX_1,HOBBIES_1_001,11101,9.99
"""
SALES_FILE, CALENDAR_FILE, PRICES_FILE = "sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"


def drop_column(csv: str, col: str) -> str:
    rows = [line.split(",") for line in csv.strip().splitlines()]
    i = rows[0].index(col)
    return "\n".join(",".join(r[:i] + r[i + 1:]) for r in rows) + "\n"


class Pipeline:
    def __init__(self, tmp_path):
        self.landing = tmp_path / "landing"
        self.raw = tmp_path / "raw" / "m5"
        self.validated = tmp_path / "validated" / "m5"
        self.contract = v.load_contract()   # the real contract, so its format is exercised too

    def deliver(self, day=DAY, force=False, **files):
        """Write the landing files (defaults + overrides) and ingest them."""
        self.landing.mkdir(exist_ok=True)
        for f in self.landing.glob("*.csv"):
            f.unlink()
        content = {SALES_FILE: SALES, CALENDAR_FILE: CALENDAR, PRICES_FILE: PRICES}
        content.update({k.replace("__", "."): t for k, t in files.items()})
        for name, text in content.items():
            (self.landing / name).write_text(text)
        status, _ = ingest_mod.ingest(day, str(self.landing), str(self.raw), force=force)
        assert status == "ingested"

    def validate(self, day=DAY):
        return v.validate(day, str(self.raw), str(self.validated), self.contract)

    def out(self, day=DAY):
        return self.validated / f"ingest_date={day.isoformat()}"

    def raw_partition(self, day=DAY):
        return self.raw / f"ingest_date={day.isoformat()}"


@pytest.fixture
def p(tmp_path):
    return Pipeline(tmp_path)


def failed_checks(report):
    return {c["check"] for s in report["stages"] for c in s["checks"] if c["failed"]}


def test_valid_partition_passes_and_writes_report(p):
    p.deliver()
    report = p.validate()

    assert report["status"] == "passed" and report["failed_stage"] is None
    assert json.loads((p.out() / "_validation.json").read_text()) == report
    manifest = (p.raw_partition() / "_manifest.json").read_bytes()
    assert report["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert report["contract_version"] == p.contract["version"]
    assert report["stats"][SALES_FILE] == {"rows": 2, "day_columns": 3}
    assert not (p.out() / "quarantine").exists()
    assert v.require_validated(DAY, str(p.raw), str(p.validated)) == report


def test_missing_manifest_raises(p):
    with pytest.raises(FileNotFoundError, match="no valid manifest"):
        p.validate()


def test_raw_file_changed_after_ingest_fails_manifest_stage(p):
    p.deliver()
    (p.raw_partition() / PRICES_FILE).write_text(PRICES.replace("2.5", "3.5"))   # same size

    report = p.validate()
    assert (report["status"], report["failed_stage"]) == ("failed", "manifest")
    with pytest.raises(v.NotValidated, match="failed validation"):
        v.require_validated(DAY, str(p.raw), str(p.validated))


def test_unparsable_csv_fails_read_stage(p):
    p.deliver(sell_prices__csv=PRICES + "CA_1,X,11101,1.0,extra\n")
    assert p.validate()["failed_stage"] == "read"


def test_missing_column_fails_structure_stage(p):
    p.deliver(sales_train_evaluation__csv=drop_column(SALES, "state_id"))
    report = p.validate()
    assert report["failed_stage"] == "structure"
    assert "required_columns" in failed_checks(report)


def test_gap_in_day_columns_fails_structure_stage(p):
    p.deliver(sales_train_evaluation__csv=drop_column(SALES, "d_2"))
    assert p.validate()["failed_stage"] == "structure"


def test_duplicate_id_fails_keys_stage(p):
    p.deliver(sales_train_evaluation__csv=SALES + SALES.splitlines()[1] + "\n")
    report = p.validate()
    assert report["failed_stage"] == "keys_volume" and "key" in failed_checks(report)


def test_bad_rows_and_cells_below_limit_are_quarantined(p):
    p.contract["defaults"]["row_error_limit"] = 0.6
    p.deliver(sell_prices__csv=PRICES.replace("9.99", "0"),
              sales_train_evaluation__csv=SALES.replace("TX,0,3,1", "TX,0,-1,1"))

    report = p.validate()
    assert report["status"] == "passed"
    assert report["quarantine"] == {"sell_prices_rows": 1, "sales_train_evaluation_cells": 1}
    rows = pd.read_parquet(p.out() / "quarantine" / "sell_prices_rows.parquet")
    assert rows[["store_id", "_failed_checks"]].to_dict("records") == [
        {"store_id": "TX_1", "_failed_checks": "range:sell_price"}]
    cells = pd.read_parquet(p.out() / "quarantine" / "sales_train_evaluation_cells.parquet")
    assert cells.to_dict("records") == [{"id": "HOBBIES_1_001_TX_1_evaluation", "d": "d_2", "value": -1}]


def test_row_failures_above_limit_escalate_to_critical(p):
    p.deliver(sell_prices__csv=PRICES.replace("9.99", "0"))   # 1 of 2 rows, limit 0.5%
    report = p.validate()
    assert report["failed_stage"] == "values"
    [check] = [c for s in report["stages"] for c in s["checks"] if c["check"] == "range:sell_price"]
    assert check["severity"] == "critical" and "escalated" in check["detail"]
    assert not (p.out() / "quarantine").exists()


def test_calendar_value_failure_is_always_critical(p):
    p.contract["defaults"]["row_error_limit"] = 0.9
    p.deliver(calendar__csv=CALENDAR.replace(",1,1,0\n", ",2,1,0\n"))   # snap_CA = 2
    report = p.validate()
    assert report["failed_stage"] == "values" and "allowed:snap_CA" in failed_checks(report)


def test_event_name_without_type_is_caught(p):
    p.contract["defaults"]["row_error_limit"] = 0.9
    p.deliver(calendar__csv=CALENDAR.replace("SuperBowl,Sporting", "SuperBowl,"))
    assert "null_together:event_name_1,event_type_1" in failed_checks(p.validate())


def test_sales_day_missing_from_calendar_fails_relationships(p):
    p.deliver(calendar__csv="".join(CALENDAR.splitlines(keepends=True)[:3]))   # d_1, d_2 only
    report = p.validate()
    assert report["failed_stage"] == "relationships"
    assert "rel:sales_days_exist_in_calendar" in failed_checks(report)


def test_orphan_price_row_is_quarantined(p):
    p.contract["defaults"]["row_error_limit"] = 0.6
    p.deliver(sell_prices__csv=PRICES + "CA_1,FOODS_1_999,11101,1.0\n")
    report = p.validate()
    assert report["status"] == "passed"
    rows = pd.read_parquet(p.out() / "quarantine" / "sell_prices_rows.parquet")
    assert rows["item_id"].tolist() == ["FOODS_1_999"]
    assert rows["_failed_checks"].tolist() == ["rel:priced_series_exist_in_sales"]


def test_volume_drop_against_previous_validated_delivery_fails(p):
    p.deliver(DAY)
    assert p.validate(DAY)["status"] == "passed"

    p.deliver(NEXT_DAY, sales_train_evaluation__csv="".join(SALES.splitlines(keepends=True)[:2]))
    report = p.validate(NEXT_DAY)
    assert report["failed_stage"] == "keys_volume" and "volume:row_drop" in failed_checks(report)


def test_volume_ignores_later_and_failed_deliveries(p):
    # A later date's report must not be used as "previous" for an earlier date.
    p.deliver(NEXT_DAY, sales_train_evaluation__csv=SALES + SALES.splitlines()[1].replace("CA_1_eval", "CA_2_eval")
              .replace(",CA_1,", ",CA_2,") + "\n")
    assert p.validate(NEXT_DAY)["status"] == "passed"
    p.deliver(DAY)
    assert p.validate(DAY)["status"] == "passed"


def test_volume_baseline_skips_failed_reports(p):
    three_series = SALES + SALES.splitlines()[1].replace("CA_1_eval", "CA_2_eval").replace(",CA_1,", ",CA_2,") + "\n"
    p.deliver(DAY, sales_train_evaluation__csv=three_series, sell_prices__csv=PRICES.replace("9.99", "0"))
    report = p.validate(DAY)
    assert report["failed_stage"] == "values" and report["stats"][SALES_FILE]["rows"] == 3

    p.deliver(NEXT_DAY)   # 2 rows: a 33% drop, but only against a delivery that failed
    assert p.validate(NEXT_DAY)["status"] == "passed"


def test_rerun_replaces_previous_output(p):
    p.contract["defaults"]["row_error_limit"] = 0.6
    p.deliver(sell_prices__csv=PRICES.replace("9.99", "0"))
    p.validate()
    assert (p.out() / "quarantine").exists()

    p.deliver(force=True)
    assert p.validate()["status"] == "passed"
    assert not (p.out() / "quarantine").exists()


def test_require_validated_rejects_partition_rebuilt_after_validation(p):
    p.deliver()
    p.validate()
    p.deliver(force=True, sell_prices__csv=PRICES.replace("9.99", "8.99"))
    with pytest.raises(v.NotValidated, match="changed since"):
        v.require_validated(DAY, str(p.raw), str(p.validated))


def test_require_validated_without_report(p):
    p.deliver()
    with pytest.raises(v.NotValidated, match="has not been validated"):
        v.require_validated(DAY, str(p.raw), str(p.validated))

def run_main(p, monkeypatch, day=DAY) -> int:
    monkeypatch.setattr(sys, "argv", ["validate", "--date", day.isoformat(),
                                      "--raw", str(p.raw), "--validated", str(p.validated)])
    with pytest.raises(SystemExit) as exc:
        v.main()
    return exc.value.code


def test_main_exits_2_when_partition_is_missing(p, monkeypatch, caplog):
    assert run_main(p, monkeypatch) == 2
    assert "no valid manifest" in caplog.text


def test_main_exits_1_when_validation_fails(p, monkeypatch, capsys):
    p.deliver(sales_train_evaluation__csv=SALES + SALES.splitlines()[1] + "\n")   # duplicate id
    assert run_main(p, monkeypatch) == 1
    assert "FAILED at stage keys_volume" in capsys.readouterr().out


def test_main_exits_0_when_validation_passes(p, monkeypatch):
    p.deliver()
    assert run_main(p, monkeypatch) == 0

    