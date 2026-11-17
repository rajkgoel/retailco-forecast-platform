"""Land M5 CSVs into a dated raw partition (see ADR-0009).

    <LANDING_URI>/*.csv  ->  <RAW_URI>/ingest_date=<yyyy-mm-dd>/*.csv + _manifest.json

Both URIs can be local paths or s3:// URIs (install the "s3" extra for S3).
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from fsspec.core import url_to_fs

REPO_ROOT = Path(__file__).resolve().parents[2]
LANDING_URI = os.environ.get("LANDING_URI", str(REPO_ROOT / "data" / "landing"))
RAW_URI = os.environ.get("RAW_URI", str(REPO_ROOT / "data" / "raw" / "m5"))
MANIFEST = "_manifest.json"
CHUNK_BYTES = 8 * 1024 * 1024

log = logging.getLogger(__name__)


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


def _sha256(fs, path: str) -> str:
    digest = hashlib.sha256()
    with fs.open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(fs, partition: str) -> dict | None:
    """Return the partition's manifest, or None if the partition is incomplete."""
    try:
        manifest = json.loads(fs.cat_file(f"{partition}/{MANIFEST}"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) and "files" in manifest else None


def _latest_complete_manifest(fs, root: str, exclude: str) -> dict | None:
    """Manifest of the most recent complete partition other than `exclude`."""
    for partition in sorted(fs.glob(f"{root}/ingest_date=*"), reverse=True):
        if partition.rstrip("/") != exclude and (manifest := _read_manifest(fs, partition)):
            return manifest
    return None


def _matches(src_fs, sources: list[str], manifest: dict) -> bool:
    """True if the landing files are byte-identical to the files recorded in `manifest`."""
    recorded = {f["name"]: f for f in manifest["files"]}
    names = {s: s.rsplit("/", 1)[-1] for s in sources}
    if sorted(recorded) != sorted(names.values()):
        return False
    # Compare sizes first so a changed delivery is usually detected without hashing.
    if any(src_fs.size(s) != recorded[n]["bytes"] for s, n in names.items()):
        return False
    return all(_sha256(src_fs, s) == recorded[n]["sha256"] for s, n in names.items())


def ingest(
    ingest_date: date,
    landing_uri: str = LANDING_URI,
    raw_uri: str = RAW_URI,
    force: bool = False,
) -> tuple[str, dict]:
    """Copy every *.csv in landing_uri into raw_uri/ingest_date=<date>/.

    Returns ("ingested", manifest), or ("skipped", manifest of the identical earlier partition).

    - A complete partition (one with a valid manifest) is immutable: re-running the same
      date fails unless force=True, which deletes and rebuilds the whole partition.
    - A partition without a manifest is left over from a crashed run and is rebuilt.
    - If the landing files are identical to the latest complete partition, nothing is copied.
    - The manifest is written last, via a temp file and rename, so a partition is complete
      only once its manifest exists. Readers must ignore partitions without one.
    """
    src_fs, src_root = url_to_fs(landing_uri)
    dest_fs, dest_root = url_to_fs(raw_uri)
    dest_root = dest_root.rstrip("/")

    sources = sorted(src_fs.glob(f"{src_root.rstrip('/')}/*.csv"))
    if not sources:
        raise FileNotFoundError(f"No CSV files found in {landing_uri}")

    partition = f"{dest_root}/ingest_date={ingest_date.isoformat()}"
    if _read_manifest(dest_fs, partition) and not force:
        raise FileExistsError(f"{partition} already exists; pass --force to overwrite")

    if not force:
        latest = _latest_complete_manifest(dest_fs, dest_root, exclude=partition)
        if latest and _matches(src_fs, sources, latest):
            return "skipped", latest

    if dest_fs.exists(partition):
        # Delete everything first: a crash from here on leaves no manifest (incomplete),
        # and no files from an earlier delivery survive into the new one.
        log.warning("removing existing partition %s before rebuilding it", partition)
        dest_fs.rm(partition, recursive=True)
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
    tmp = f"{partition}/{MANIFEST}.tmp"
    with dest_fs.open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    dest_fs.mv(tmp, f"{partition}/{MANIFEST}")
    return "ingested", manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=utc_today(),
                        help="ingest date, YYYY-MM-DD (default: today in UTC)")
    parser.add_argument("--landing", default=LANDING_URI, help="landing location (env: LANDING_URI)")
    parser.add_argument("--raw", default=RAW_URI, help="raw layer root (env: RAW_URI)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing partition")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        status, manifest = ingest(args.date, args.landing, args.raw, args.force)
    except (FileExistsError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(2)
    if status == "skipped":
        print(f"landing files are identical to ingest_date={manifest['ingest_date']}; nothing to do")
        return
    for f in manifest["files"]:
        print(f"copied {f['name']} ({f['bytes']:,} bytes, sha256 {f['sha256'][:12]})")
    print(f"partition ingest_date={manifest['ingest_date']} written to {args.raw}")


if __name__ == "__main__":
    main()
