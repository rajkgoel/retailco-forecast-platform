"""Validate a raw M5 partition against contracts/data-contract.yaml (ADR-0004).

    <RAW_URI>/ingest_date=<d>/  ->  <VALIDATED_URI>/ingest_date=<d>/_validation.json
                                                                 /quarantine/*.parquet

Stages run in order; the run stops after the first stage with a critical failure.
All checks within a stage run, so one run reports every problem in that stage.

    manifest -> read -> structure -> keys_volume -> values -> relationships -> quarantine

Critical failures fail the run. Row-level failures are quarantined, unless they exceed
`row_error_limit` of a file, in which case they escalate to critical.

The report is written last (temp file + rename) whether the run passes or fails, and
records the sha256 of the raw manifest it validated. Downstream steps call
`require_validated()`, which also detects a raw partition rebuilt after validation.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fsspec.core import url_to_fs

from src.ingest import ingest as ingest_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "data-contract.yaml"
RAW_URI = ingest_mod.RAW_URI
VALIDATED_URI = os.environ.get("VALIDATED_URI", str(REPO_ROOT / "data" / "validated" / "m5"))
REPORT = "_validation.json"
CRITICAL, ROW = "critical", "row"
DAY_COLUMNS = "day_columns"  # relationship keyword: "every day column of the file"

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    check: str        # e.g. "required_columns", "allowed:store_id"
    file: str | None
    severity: str     # "critical" or "row"
    failed: int       # rows / cells / items that failed
    total: int
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.failed == 0


class StageFailed(Exception):
    pass


class NotValidated(Exception):
    """Raised by require_validated() when a partition has no current, passing report."""


# --------------------------------------------------------------------------- policy

def run_stage(stage: str, results: list[CheckResult], report: dict) -> None:
    """Record every result for the stage, then stop if any critical check failed."""
    report["stages"].append({"stage": stage, "checks": [asdict(r) for r in results]})
    if any(r.severity == CRITICAL and not r.ok for r in results):
        raise StageFailed(stage)


def escalate(r: CheckResult, limit: float) -> CheckResult:
    """ADR-0004: a row-level check failing on more than `limit` of a file becomes critical."""
    if r.severity == ROW and r.total and r.failed / r.total > limit:
        r.severity = CRITICAL
        r.detail = f"{r.detail} ({r.failed / r.total:.2%} > {limit:.1%}, escalated)".strip()
    return r


def apply_policy(r: CheckResult, file_spec: dict, limit: float) -> CheckResult:
    """A file marked `severity: critical` never quarantines; otherwise escalate by volume."""
    if file_spec.get("severity") == CRITICAL:
        r.severity = CRITICAL
        return r
    return escalate(r, limit)


# --------------------------------------------------------------------------- stage 1: manifest

def check_manifest(fs, partition: str, manifest: dict, contract: dict) -> list[CheckResult]:
    listed = {f["name"]: f for f in manifest["files"]}
    results = []
    for name in contract["files"]:
        if name not in listed:
            results.append(CheckResult("in_manifest", name, CRITICAL, 1, 1, "required by contract, not in manifest"))
            continue
        f, path = listed[name], f"{partition}/{name}"
        ok = (fs.exists(path)
              and fs.size(path) == f["bytes"]                       # cheap check first
              and ingest_mod._sha256(fs, path) == f["sha256"])
        results.append(CheckResult("matches_manifest", name, CRITICAL, 0 if ok else 1, 1,
                                   "" if ok else "missing, or size/sha256 differs from manifest"))
    return results


# --------------------------------------------------------------------------- reading

def read_file(fs, path: str, spec: dict) -> pd.DataFrame:
    """Read a CSV; string and date columns as "string", everything else inferred."""
    dtype = {c: "string" for c, s in spec["columns"].items() if s["type"] in ("string", "date")}
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, dtype=dtype)


# --------------------------------------------------------------------------- stage 3: structure

def _sequence_problems(nums: list[int], start: int) -> list[str]:
    """Problems with a sequence that should be exactly start, start+1, ... in order."""
    if not nums:
        return ["no values"]
    problems = []
    if min(nums) != start:
        problems.append(f"starts at {min(nums)}, expected {start}")
    gaps = sorted(set(range(start, max(nums) + 1)) - set(nums))
    if gaps:
        problems.append(f"{len(gaps)} gap(s), first: {gaps[:5]}")
    if len(nums) != len(set(nums)):
        problems.append(f"{len(nums) - len(set(nums))} duplicate(s)")
    if nums != sorted(nums):
        problems.append("not in increasing order")
    return problems


def check_required_columns(name: str, df: pd.DataFrame, spec: dict) -> CheckResult:
    expected = list(spec["columns"])
    missing = [c for c in expected if c not in df.columns]
    return CheckResult("required_columns", name, CRITICAL, len(missing), len(expected),
                       f"missing: {', '.join(missing)}" if missing else "")


def check_day_columns(name: str, df: pd.DataFrame, spec: dict) -> tuple[CheckResult, list[str]]:
    """Day columns must be d_<start>..d_<N> with no gaps, repeats or reordering.

    Returns the result and the day column names in numeric order.
    """
    dc = spec[DAY_COLUMNS]
    regex = re.compile(dc["pattern"])
    day_cols = [c for c in df.columns if regex.fullmatch(c)]
    nums = [int(regex.fullmatch(c).group(1)) for c in day_cols]
    problems = _sequence_problems(nums, dc["contiguous_from"])
    # pandas renames a repeated header "d_5" to "d_5.1", which then no longer matches.
    mangled = [c for c in df.columns
               if not regex.fullmatch(c) and regex.fullmatch(re.sub(r"\.\d+$", "", c))]
    if mangled:
        problems.append(f"repeated header(s): {mangled[:5]}")
    ordered = sorted(day_cols, key=lambda c: int(regex.fullmatch(c).group(1)))
    return CheckResult(DAY_COLUMNS, name, CRITICAL, len(problems), len(nums), "; ".join(problems)), ordered


def _is_float(s: pd.Series) -> bool:
    # A float column whose values are all whole numbers is read as int; that is fine.
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)


_TYPE_TESTS = {
    "int": pd.api.types.is_integer_dtype,
    "float": _is_float,
    "string": pd.api.types.is_string_dtype,
    "date": pd.api.types.is_string_dtype,   # parsed by check_date_contiguous
}


def _type_test(type_name: str):
    try:
        return _TYPE_TESTS[type_name]
    except KeyError:
        raise ValueError(f"contract uses unknown type {type_name!r}; known: {sorted(_TYPE_TESTS)}") from None


def check_types(name: str, df: pd.DataFrame, spec: dict, day_cols: list[str],
                default_nullable: bool = False) -> list[CheckResult]:
    """One critical result per column (decided per column, not per cell)."""
    results = []
    for col, cs in spec["columns"].items():
        if col not in df.columns:
            continue  # reported by check_required_columns
        s = df[col]
        nullable = cs.get("nullable", default_nullable)
        # An entirely empty column is read as float64 whatever its declared type.
        ok = _type_test(cs["type"])(s) or (nullable and s.isna().all())
        results.append(CheckResult(f"type:{col}", name, CRITICAL, 0 if ok else 1, 1,
                                   "" if ok else f"expected {cs['type']}, got {s.dtype}"))
    if day_cols:
        expected = spec[DAY_COLUMNS]["type"]
        test = _type_test(expected)
        bad = [c for c in day_cols if not test(df[c])]
        results.append(CheckResult(f"type:{DAY_COLUMNS}", name, CRITICAL, len(bad), len(day_cols),
                                   f"expected {expected}: {bad[:5]}" if bad else ""))
    return results


# --------------------------------------------------------------------------- stage 4: keys and volume

def check_key(name: str, df: pd.DataFrame, key: list[str], check: str = "key") -> CheckResult:
    """Key must be unique and never null. Also used for `unique: true` columns."""
    dup = df.duplicated(subset=key, keep=False)       # keep=False marks every copy
    null = df[key].isna().any(axis=1)
    bad = dup | null
    sample = df.loc[bad, key].drop_duplicates().head(3).to_dict("records")
    return CheckResult(check, name, CRITICAL, int(bad.sum()), len(df), f"e.g. {sample}" if sample else "")


def check_sequence(name: str, df: pd.DataFrame, col: str, pattern: str, start: int) -> CheckResult:
    """Values like d_1, d_2, ... (any row order) must form start..N with no gaps or repeats."""
    regex = re.compile(pattern)
    values = df[col].dropna().tolist()
    matches = [regex.fullmatch(v) for v in values]
    unmatched = [v for v, m in zip(values, matches) if not m]
    problems = _sequence_problems(sorted(int(m.group(1)) for m in matches if m), start)
    if unmatched:
        problems.append(f"{len(unmatched)} value(s) not matching {pattern}: {unmatched[:5]}")
    return CheckResult(f"sequence:{col}", name, CRITICAL, len(problems), len(df), "; ".join(problems))


def check_date_contiguous(name: str, df: pd.DataFrame, col: str, fmt: str) -> CheckResult:
    """Dates must parse and step exactly one day (gaps and repeats both fail)."""
    parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
    unparsable = int((parsed.isna() & df[col].notna()).sum())
    steps = parsed.dropna().sort_values().diff().dropna()
    bad_steps = int((steps != pd.Timedelta(days=1)).sum())
    parts = [f"{unparsable} unparsable (format {fmt})"] if unparsable else []
    if bad_steps:
        parts.append(f"{bad_steps} step(s) that are not one day (gap or repeat)")
    return CheckResult(f"contiguous:{col}", name, CRITICAL, unparsable + bad_steps, len(df), "; ".join(parts))


def check_volume(name: str, current: dict, previous: dict | None, spec: dict) -> list[CheckResult]:
    """Compare {"rows", "day_columns"} with the previous validated delivery of this file."""
    if previous is None:
        return [CheckResult("volume", name, CRITICAL, 0, current["rows"], "no previous validated delivery")]
    vol = spec.get("volume", {})
    prev_rows, rows = previous["rows"], current["rows"]
    results = []
    if "max_row_drop_vs_previous" in vol:
        limit = vol["max_row_drop_vs_previous"]
        drop = (prev_rows - rows) / prev_rows if prev_rows else 0.0
        results.append(CheckResult("volume:row_drop", name, CRITICAL, int(drop > limit), 1,
                                   f"rows {prev_rows} -> {rows} ({drop:.1%} drop, limit {limit:.0%})"))
    if vol.get("not_fewer_than_previous"):
        results.append(CheckResult("volume:rows", name, CRITICAL, int(rows < prev_rows), 1,
                                   f"rows {prev_rows} -> {rows}"))
    if spec.get(DAY_COLUMNS, {}).get("not_fewer_than_previous"):
        prev_days, days = previous["day_columns"], current["day_columns"]
        results.append(CheckResult("volume:day_columns", name, CRITICAL, int(days < prev_days), 1,
                                   f"day columns {prev_days} -> {days}"))
    return results


# --------------------------------------------------------------------------- stage 5: values
# Each returns (result, mask): mask is a boolean Series aligned to df.index, True = bad row.
# Nulls are only check_nulls' concern, so a null value is never counted twice.

def _none(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def check_nulls(name: str, df: pd.DataFrame, col: str, nullable: bool) -> tuple[CheckResult, pd.Series]:
    bad = _none(df) if nullable else df[col].isna()
    return CheckResult(f"not_null:{col}", name, ROW, int(bad.sum()), len(df)), bad


def check_allowed(name: str, df: pd.DataFrame, col: str, allowed: list) -> tuple[CheckResult, pd.Series]:
    bad = df[col].notna() & ~df[col].isin(allowed)
    return CheckResult(f"allowed:{col}", name, ROW, int(bad.sum()), len(df)), bad


def check_range(name: str, df: pd.DataFrame, col: str, min_=None, max_=None) -> tuple[CheckResult, pd.Series]:
    s, bad = df[col], _none(df)
    if min_ is not None:
        bad |= s < min_
    if max_ is not None:
        bad |= s > max_
    bad = (bad & s.notna()).fillna(False).astype(bool)
    return CheckResult(f"range:{col}", name, ROW, int(bad.sum()), len(df), f"allowed [{min_}, {max_}]"), bad


def check_pattern(name: str, df: pd.DataFrame, col: str, pattern: str) -> tuple[CheckResult, pd.Series]:
    s = df[col]
    matched = s.str.fullmatch(pattern).fillna(False).astype(bool)
    bad = s.notna() & ~matched
    return CheckResult(f"pattern:{col}", name, ROW, int(bad.sum()), len(df), pattern), bad


def check_null_together(name: str, df: pd.DataFrame, a: str, b: str) -> tuple[CheckResult, pd.Series]:
    bad = df[a].isna() != df[b].isna()
    return CheckResult(f"null_together:{a},{b}", name, ROW, int(bad.sum()), len(df)), bad


def check_day_values(name: str, df: pd.DataFrame, day_cols: list[str], spec: dict) -> tuple[CheckResult, pd.DataFrame]:
    """Cell-level check: returns the bad (id, d, value) cells instead of a row mask.

    Uses numpy rather than df.stack(): on the real file stack() builds a 59M-row index.
    """
    id_col = spec["key"][0]
    values = df[day_cols].to_numpy()
    bad = values < spec[DAY_COLUMNS]["min"]
    r, c = np.nonzero(bad)
    cells = pd.DataFrame({
        id_col: df[id_col].to_numpy()[r],
        "d": np.asarray(day_cols, dtype=object)[c],
        "value": values[r, c],
    })
    return CheckResult(f"min:{DAY_COLUMNS}", name, ROW, len(cells), int(bad.size),
                       f"min {spec[DAY_COLUMNS]['min']}"), cells


# --------------------------------------------------------------------------- stage 6: relationships

def check_relationship(rel: dict, frames: dict[str, pd.DataFrame],
                       day_cols: list[str]) -> tuple[CheckResult, pd.Series | None]:
    """Every `from` value must exist in `to`. Mask is aligned to the `from` file (None for day columns)."""
    src, dst = rel["from"], rel["to"]
    severity = rel.get("severity", ROW)
    check = f"rel:{rel['name']}"
    if src["columns"] == DAY_COLUMNS:
        known = set(frames[dst["file"]][dst["columns"][0]].dropna())
        missing = [d for d in day_cols if d not in known]
        return CheckResult(check, src["file"], severity, len(missing), len(day_cols),
                           f"not in {dst['file']}: {missing[:5]}" if missing else ""), None

    left = frames[src["file"]][src["columns"]]
    # drop_duplicates is required: a repeated key on the right would make the left
    # merge return extra rows, and the mask would no longer line up with `left`.
    keys = frames[dst["file"]][dst["columns"]].drop_duplicates().set_axis(src["columns"], axis=1)
    merged = left.merge(keys, how="left", on=src["columns"], indicator=True)
    bad = pd.Series((merged["_merge"] == "left_only").to_numpy(), index=left.index)
    return CheckResult(check, src["file"], severity, int(bad.sum()), len(left),
                       f"{src['columns']} not in {dst['file']}"), bad


# --------------------------------------------------------------------------- orchestration

def load_contract(path: str | Path = CONTRACT) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_report(fs, partition: str) -> dict | None:
    try:
        return json.loads(fs.cat_file(f"{partition}/{REPORT}"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _previous_stats(fs, root: str, ingest_date: date) -> dict | None:
    """Stats of the latest *passed* validation for an ingest date before this one."""
    for partition in sorted(fs.glob(f"{root}/ingest_date=*"), reverse=True):
        if partition.rstrip("/").rsplit("ingest_date=", 1)[-1] >= ingest_date.isoformat():
            continue
        report = _read_report(fs, partition)
        if report and report.get("status") == "passed":
            return report["stats"]
    return None


def _manifest_sha256(fs, partition: str) -> str:
    return hashlib.sha256(fs.cat_file(f"{partition}/{ingest_mod.MANIFEST}")).hexdigest()


def _quarantine_rows(df: pd.DataFrame, flags: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    """Rows flagged by any check, with a `_failed_checks` column naming the checks."""
    masks = [(check, m.to_numpy(dtype=bool)) for check, m in flags]
    bad = np.zeros(len(df), dtype=bool)
    for _, m in masks:
        bad |= m
    idx = np.flatnonzero(bad)
    reasons = [[] for _ in idx]
    for check, m in masks:
        for j in np.flatnonzero(m[idx]):
            reasons[j].append(check)
    out = df.iloc[idx].copy()
    out["_failed_checks"] = [";".join(r) for r in reasons]
    return out


def _write_outputs(fs, out: str, report: dict, quarantine: dict[str, pd.DataFrame]) -> None:
    """Replace the validated partition; the report is written last, via temp + rename."""
    if fs.exists(out):
        fs.rm(out, recursive=True)
    fs.makedirs(f"{out}/quarantine" if quarantine else out, exist_ok=True)
    for qname, q in quarantine.items():
        with fs.open(f"{out}/quarantine/{qname}.parquet", "wb") as f:
            q.to_parquet(f, index=False)
    tmp = f"{out}/{REPORT}.tmp"
    with fs.open(tmp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    fs.mv(tmp, f"{out}/{REPORT}")


def validate(ingest_date: date, raw_uri: str = RAW_URI, validated_uri: str = VALIDATED_URI,
             contract: dict | None = None) -> dict:
    """Validate raw_uri/ingest_date=<d>/ and write validated_uri/ingest_date=<d>/. Returns the report."""
    contract = contract or load_contract()
    raw_fs, raw_root = url_to_fs(raw_uri)
    out_fs, out_root = url_to_fs(validated_uri)
    raw_root, out_root = raw_root.rstrip("/"), out_root.rstrip("/")
    tag = f"ingest_date={ingest_date.isoformat()}"
    partition, out = f"{raw_root}/{tag}", f"{out_root}/{tag}"

    manifest = ingest_mod._read_manifest(raw_fs, partition)
    if manifest is None:
        raise FileNotFoundError(f"{partition} has no valid manifest; ingest it first")

    specs = contract["files"]
    defaults = contract.get("defaults", {})
    limit = defaults.get("row_error_limit", 0.0)
    default_nullable = defaults.get("nullable", False)
    previous = _previous_stats(out_fs, out_root, ingest_date) or {}

    report = {
        "dataset": contract.get("dataset"),
        "ingest_date": ingest_date.isoformat(),
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_version": contract.get("version"),
        "raw_partition": partition,
        "manifest_sha256": _manifest_sha256(raw_fs, partition),
        "status": None,
        "failed_stage": None,
        "stats": {},
        "quarantine": {},
        "stages": [],
    }
    frames: dict[str, pd.DataFrame] = {}
    day_cols: dict[str, list[str]] = {}
    row_flags: dict[str, list[tuple[str, pd.Series]]] = {n: [] for n in specs}
    cells: dict[str, pd.DataFrame] = {}

    try:
        run_stage("manifest", check_manifest(raw_fs, partition, manifest, contract), report)

        results = []
        for n, s in specs.items():
            try:
                frames[n] = read_file(raw_fs, f"{partition}/{n}", s)
                results.append(CheckResult("readable", n, CRITICAL, 0, 1))
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
                results.append(CheckResult("readable", n, CRITICAL, 1, 1, str(e).strip()[:300]))
        run_stage("read", results, report)

        results = []
        for n, s in specs.items():
            df = frames[n]
            results.append(check_required_columns(n, df, s))
            if DAY_COLUMNS in s:
                r, day_cols[n] = check_day_columns(n, df, s)
                results.append(r)
            results += check_types(n, df, s, day_cols.get(n, []), default_nullable)
        run_stage("structure", results, report)
        report["stats"] = {n: {"rows": len(df), "day_columns": len(day_cols.get(n, []))}
                           for n, df in frames.items()}

        results = []
        for n, s in specs.items():
            df = frames[n]
            results.append(check_key(n, df, s["key"]))
            for col, cs in s["columns"].items():
                if cs.get("unique"):
                    results.append(check_key(n, df, [col], f"unique:{col}"))
                if "contiguous_from" in cs:
                    results.append(check_sequence(n, df, col, cs["pattern"], cs["contiguous_from"]))
                if cs.get("contiguous") == "daily":
                    results.append(check_date_contiguous(n, df, col, cs["format"]))
            results += check_volume(n, report["stats"][n], previous.get(n), s)
        run_stage("keys_volume", results, report)

        results = []
        for n, s in specs.items():
            df = frames[n]
            checks = []
            for col, cs in s["columns"].items():
                checks.append(check_nulls(n, df, col, cs.get("nullable", default_nullable)))
                if "allowed" in cs:
                    checks.append(check_allowed(n, df, col, cs["allowed"]))
                if "min" in cs or "max" in cs:
                    checks.append(check_range(n, df, col, cs.get("min"), cs.get("max")))
                if "pattern" in cs:
                    checks.append(check_pattern(n, df, col, cs["pattern"]))
            for a, b in s.get("null_together", []):
                checks.append(check_null_together(n, df, a, b))
            for r, mask in checks:
                results.append(apply_policy(r, s, limit))
                if not r.ok:
                    row_flags[n].append((r.check, mask))
            if day_cols.get(n) and "min" in s[DAY_COLUMNS]:
                r, cells[n] = check_day_values(n, df, day_cols[n], s)
                results.append(apply_policy(r, s, limit))
        run_stage("values", results, report)

        results = []
        for rel in contract.get("relationships", []):
            r, mask = check_relationship(rel, frames, day_cols.get(rel["from"]["file"], []))
            results.append(apply_policy(r, specs[r.file], limit))
            if mask is not None and not r.ok:
                row_flags[r.file].append((r.check, mask))
        run_stage("relationships", results, report)

        # Several checks can each stay under the limit while together quarantining too much.
        quarantine = {}
        results = []
        for n, s in specs.items():
            rows = _quarantine_rows(frames[n], row_flags[n])
            results.append(apply_policy(CheckResult("quarantined_rows", n, ROW, len(rows), len(frames[n])), s, limit))
            if len(rows):
                quarantine[f"{Path(n).stem}_rows"] = rows
            if n in cells and len(cells[n]):
                quarantine[f"{Path(n).stem}_cells"] = cells[n]
        run_stage("quarantine", results, report)

    except StageFailed as e:
        report["status"], report["failed_stage"] = "failed", str(e)
        quarantine = {}   # a failed run quarantines nothing; the whole partition is rejected
    else:
        report["status"] = "passed"
        report["quarantine"] = {q: len(df) for q, df in quarantine.items()}

    _write_outputs(out_fs, out, report, quarantine)
    return report


def require_validated(ingest_date: date, raw_uri: str = RAW_URI, validated_uri: str = VALIDATED_URI) -> dict:
    """Return the passing report for ingest_date, or raise NotValidated.

    Also fails if the raw partition was rebuilt (ingest --force) after validation.
    """
    raw_fs, raw_root = url_to_fs(raw_uri)
    out_fs, out_root = url_to_fs(validated_uri)
    tag = f"ingest_date={ingest_date.isoformat()}"
    report = _read_report(out_fs, f"{out_root.rstrip('/')}/{tag}")
    if report is None:
        raise NotValidated(f"{tag} has not been validated")
    if report.get("status") != "passed":
        raise NotValidated(f"{tag} failed validation at stage {report.get('failed_stage')}")
    partition = f"{raw_root.rstrip('/')}/{tag}"
    try:
        current = _manifest_sha256(raw_fs, partition)
    except FileNotFoundError:
        raise NotValidated(f"{partition} has no manifest") from None
    if current != report["manifest_sha256"]:
        raise NotValidated(f"{partition} changed since it was validated; re-run validate")
    return report


def _latest_raw_date(raw_uri: str) -> date:
    fs, root = url_to_fs(raw_uri)
    manifest = ingest_mod._latest_complete_manifest(fs, root.rstrip("/"), exclude="")
    if manifest is None:
        raise FileNotFoundError(f"no complete raw partition in {raw_uri}")
    return date.fromisoformat(manifest["ingest_date"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat,
                        help="ingest date, YYYY-MM-DD (default: latest complete raw partition)")
    parser.add_argument("--raw", default=RAW_URI, help="raw layer root (env: RAW_URI)")
    parser.add_argument("--validated", default=VALIDATED_URI, help="validated layer root (env: VALIDATED_URI)")
    parser.add_argument("--contract", default=str(CONTRACT), help="data contract YAML")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        ingest_date = args.date or _latest_raw_date(args.raw)
        report = validate(ingest_date, args.raw, args.validated, load_contract(args.contract))
    except FileNotFoundError as e:
        log.error("%s", e)
        sys.exit(2)

    for stage in report["stages"]:
        for c in stage["checks"]:
            if c["failed"]:
                print(f"[{stage['stage']}] {c['severity']:8} {c['file']} {c['check']}: "
                      f"{c['failed']}/{c['total']} {c['detail']}")
    for q, n in report["quarantine"].items():
        print(f"quarantined {n:,} in {q}")
    print(f"ingest_date={report['ingest_date']}: {report['status'].upper()}"
          + (f" at stage {report['failed_stage']}" if report["failed_stage"] else ""))
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
