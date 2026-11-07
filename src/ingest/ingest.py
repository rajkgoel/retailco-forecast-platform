"""Land M5 CSVs into a dated raw partition (see ADR-0009).

    <LANDING_URI>/*.csv  ->  <RAW_URI>/ingest_date=<yyyy-mm-dd>/*.csv + _manifest.json

Both URIs can be local paths or s3:// URIs (install the "s3" extra for S3).
"""

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

from fsspec.core import url_to_fs

REPO_ROOT = Path(__file__).resolve().parents[2]
LANDING_URI = os.environ.get("LANDING_URI", str(REPO_ROOT / "data" / "landing"))
RAW_URI = os.environ.get("RAW_URI", str(REPO_ROOT / "data" / "raw" / "m5"))
MANIFEST = "_manifest.json"
CHUNK_BYTES = 8 * 1024 * 1024


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _copy_with_hash(src_fs, src: str, dest_fs, dest: str) -> tuple[int, str]:
    """Stream src to dest (works across filesystems) and return (bytes, sha256)."""
    digest, size = hashlib.sha256(), 0
    with src_fs.open(src, "rb") as fin, dest_fs.open(dest, "wb") as fout:
        while chunk := fin.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            fout.write(chunk)
    return size, digest.hexdigest()


def ingest(
    ingest_date: date,
    landing_uri: str = LANDING_URI,
    raw_uri: str = RAW_URI,
    force: bool = False,
) -> dict:
    """Copy every *.csv in landing_uri into raw_uri/ingest_date=<date>/ and return the manifest.

    Raw partitions are immutable: an existing partition is never overwritten unless
    force=True. The manifest is written last, so a partition without one is incomplete.
    """
    src_fs, src_root = url_to_fs(landing_uri)
    dest_fs, dest_root = url_to_fs(raw_uri)

    sources = sorted(src_fs.glob(f"{src_root.rstrip('/')}/*.csv"))
    if not sources:
        raise FileNotFoundError(f"No CSV files found in {landing_uri}")

    partition = f"{dest_root.rstrip('/')}/ingest_date={ingest_date.isoformat()}"
    if dest_fs.exists(partition) and dest_fs.ls(partition) and not force:
        raise FileExistsError(f"{partition} already exists; pass --force to overwrite")
    dest_fs.makedirs(partition, exist_ok=True)

    files = []
    for src in sources:
        name = src.rsplit("/", 1)[-1]
        size, sha256 = _copy_with_hash(src_fs, src, dest_fs, f"{partition}/{name}")
        files.append({"name": name, "bytes": size, "sha256": sha256})

    manifest = {
        "dataset": "m5",
        "ingest_date": ingest_date.isoformat(),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": landing_uri,
        "files": files,
    }
    with dest_fs.open(f"{partition}/{MANIFEST}", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=utc_today(),
                        help="ingest date, YYYY-MM-DD (default: today in UTC)")
    parser.add_argument("--landing", default=LANDING_URI, help="landing location (env: LANDING_URI)")
    parser.add_argument("--raw", default=RAW_URI, help="raw layer root (env: RAW_URI)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing partition")
    args = parser.parse_args()

    manifest = ingest(args.date, args.landing, args.raw, args.force)
    for f in manifest["files"]:
        print(f"copied {f['name']} ({f['bytes']:,} bytes, sha256 {f['sha256'][:12]})")
    print(f"partition ingest_date={manifest['ingest_date']} written to {args.raw}")


if __name__ == "__main__":
    main()
