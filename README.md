# RetailCo Forecast Platform

A daily sales-forecasting pipeline for RetailCo, built on the M5 (Walmart) dataset: 10 stores, 3,049 products, 1,941 days of sales.

Pipeline steps, each a separate command:

| Step | Status | Reads | Writes |
|---|---|---|---|
| **ingest** | done | `data/landing/*.csv` | `data/raw/m5/ingest_date=YYYY-MM-DD/` |
| **validate** | done | the raw partition | `data/validated/m5/ingest_date=YYYY-MM-DD/` |
| curate | planned | the validated partition | `data/curated/` |
| aggregate | planned | curated | `data/aggregates/` |
| baseline + score | planned | aggregates | eval report |

Design decisions are recorded as ADRs in the `architect-notebook` repo. ADR-0004 covers data contracts and ADR-0009 the raw layer layout.

## Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- `make` (optional). Every `make` target below also shows the plain `uv` command it runs, which works on Windows without `make`.

## Setup

```bash
uv sync                 # create .venv and install dependencies, including dev tools (pytest)
uv sync --extra s3      # also install s3fs, if you read or write s3:// paths
```

### Get the data

The data is not in git (`data/` is ignored). Download the M5 files from the Kaggle competition "M5 Forecasting - Accuracy" and put these three in `data/landing/`:

```
data/landing/calendar.csv
data/landing/sales_train_evaluation.csv
data/landing/sell_prices.csv
```

## Running the pipeline

```bash
make ingest                      # uv run python -m src.ingest.ingest
make validate                    # uv run python -m src.validate.validate
```

Both take an optional date:

```bash
make ingest DATE=2026-10-30      # uv run python -m src.ingest.ingest --date 2026-10-30
make validate DATE=2026-10-30    # uv run python -m src.validate.validate --date 2026-10-30
```

Add `--help` to either module to see all of its options.

### ingest

Copies every `*.csv` in landing into a dated raw partition and writes `_manifest.json` (file names, sizes, SHA-256) last. A partition with no manifest is incomplete and is ignored by every reader.

| Option | Default | Meaning |
|---|---|---|
| `--date YYYY-MM-DD` | today (UTC) | ingest date of the partition |
| `--landing URI` | `data/landing` (env `LANDING_URI`) | where to read CSVs from |
| `--raw URI` | `data/raw/m5` (env `RAW_URI`) | raw layer root |
| `--force` | off | rebuild a partition that is already complete |

Rerun behaviour:
- **Same date, complete partition:** fails unless you pass `--force`.
- **Same date, crashed partition (no manifest):** rebuilt without `--force`.
- **New date, landing identical to the latest complete partition:** skipped; nothing is copied.
- **`--force`:** the partition is deleted first, then rebuilt.

### validate

Checks a raw partition against [`contracts/data-contract.yaml`](contracts/data-contract.yaml) in stages: manifest, read, structure, keys and volume, values, relationships, and quarantine total. The run stops after the first stage with a critical failure.

| Option | Default | Meaning |
|---|---|---|
| `--date YYYY-MM-DD` | latest complete raw partition | partition to validate |
| `--raw URI` | `data/raw/m5` (env `RAW_URI`) | raw layer root |
| `--validated URI` | `data/validated/m5` (env `VALIDATED_URI`) | where results go |
| `--contract PATH` | `contracts/data-contract.yaml` | data contract |

Output, written for every run whether it passes or fails:

```
data/validated/m5/ingest_date=YYYY-MM-DD/
    _validation.json            # status, every check, row counts, contract version, raw manifest hash
    quarantine/*.parquet        # only when the run passes and some rows or cells were quarantined
```

- **Exit codes** (ingest and validate): `0` success, `1` validation ran and rejected the data, `2` could not run (partition already exists, no manifest, missing file). Retry `2` after fixing the setup; alert on `1`.
- **Severity:** row-level failures above `row_error_limit` (0.5%) of a file are escalated to critical. Any failure in `calendar.csv` is always critical.
- **Downstream steps** must call `require_validated()`. It rejects a partition that failed validation, was never validated, or was rebuilt with `ingest --force` afterwards.
- **Contract changes:** bump `version` in the contract whenever you change a rule.

### Using S3

Every location option accepts an `s3://` URI once the `s3` extra is installed. Credentials come from the standard AWS environment variables or profile.

```bash
LANDING_URI=s3://bucket/landing RAW_URI=s3://bucket/raw/m5 uv run python -m src.ingest.ingest
```

## Tests

```bash
make test                                              # uv run pytest
uv run pytest -q                                       # quiet output
uv run pytest tests/validate                           # one folder
uv run pytest tests/ingest/ingest_test.py              # one file
uv run pytest -k "quarantine"                          # tests whose names match
uv run pytest tests/validate/validate_test.py::TestKey # one class
uv run pytest -x                                       # stop at the first failure
```

The tests create their own small CSV files in temporary folders. They don't need the M5 data and never touch `data/`.

## Project layout

```
contracts/data-contract.yaml   rules validate checks the raw data against
src/ingest/ingest.py           landing -> raw partition + manifest
src/validate/validate.py       raw partition -> validation report + quarantine
tests/                         pytest tests, one folder per step
data/                          landing, raw, validated, curated, aggregates (git-ignored)
Makefile                       shortcuts for the commands above
```
