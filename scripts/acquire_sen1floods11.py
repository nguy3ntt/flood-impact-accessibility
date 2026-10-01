"""Acquire explicitly named public Sen1Floods11 pilot objects with a byte cap.

This script records bytes and hashes, but does not grant redistribution rights.
It never lists and downloads a whole prefix implicitly.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen


BUCKET = "sen1floods11"
API_ROOT = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
RAW_ROOT = Path("data/raw/sen1floods11")
MANIFEST = Path(".project/m1_source_inventory.json")


def valid_key(key: str) -> bool:
    parts = PurePosixPath(key).parts
    return (
        key.startswith("v1.1/")
        and len(parts) >= 2
        and not key.startswith("/")
        and "\\" not in key
        and all(part not in ("", ".", "..") for part in key.split("/"))
    )


def object_info(key: str) -> dict:
    url = API_ROOT + quote(key, safe="")
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def acquire(key: str, max_bytes: int) -> dict:
    if not valid_key(key):
        raise ValueError(f"Invalid object key: {key!r}")
    if max_bytes <= 0:
        raise ValueError("Remaining byte cap must be positive")
    info = object_info(key)
    expected = int(info["size"])
    if expected < 0 or expected > max_bytes:
        raise ValueError(f"Object is {expected} bytes; cap is {max_bytes}")
    target = RAW_ROOT.joinpath(*key.split("/"))
    if target.exists():
        raise FileExistsError(f"Raw object already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Partial file exists: {temporary}")
    source_url = f"https://storage.googleapis.com/{BUCKET}/{quote(key, safe='/')}"
    download_url = source_url + "?generation=" + str(info["generation"])
    digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    received = 0
    try:
        with urlopen(download_url, timeout=60) as source, temporary.open("xb") as sink:
            while chunk := source.read(min(1024 * 1024, max_bytes - received + 1)):
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError(f"Download exceeded cap of {max_bytes} bytes")
                sink.write(chunk)
                digest.update(chunk)
                md5.update(chunk)
        if received != expected:
            raise ValueError(f"Expected {expected} bytes, received {received}")
        expected_md5 = info.get("md5Hash")
        if expected_md5 and base64.b64encode(md5.digest()).decode() != expected_md5:
            raise ValueError("Downloaded bytes differ from the bucket MD5")
        temporary.rename(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source_id": f"sen1floods11:{key}",
        "dataset_version": "v1.1",
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_filename": Path(key).name,
        "local_path": target.as_posix(),
        "bytes": received,
        "sha256": digest.hexdigest(),
        "generation": info.get("generation"),
        "bucket_md5_base64": info.get("md5Hash"),
        "updated_utc": info.get("updated"),
        "content_type": info.get("contentType"),
        "licence_url": None,
        "licence_status": "unresolved; internal feasibility only",
        "redistribution_status": "prohibited pending explicit rights review",
        "attribution": "Cloud to Street; Bonafilia et al. (2020)",
        "coverage": None,
        "notes": "Exact public object; no downstream geographic claim until audited.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", nargs="+", help="Exact v1.1 object keys, never prefixes")
    parser.add_argument("--max-total-bytes", type=int, default=250 * 1024 * 1024)
    args = parser.parse_args()
    if args.max_total_bytes <= 0:
        parser.error("Byte cap must be positive")
    records = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []
    used = sum(record["bytes"] for record in records)
    for key in args.key:
        remaining = args.max_total_bytes - used
        if remaining <= 0:
            raise ValueError("Pilot byte cap reached")
        record = acquire(key, remaining)
        records.append(record)
        used += record["bytes"]
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        temporary = MANIFEST.with_name(MANIFEST.name + ".part")
        temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        temporary.replace(MANIFEST)
        print(f"{record['local_path']}: {record['bytes']} bytes, sha256={record['sha256']}")
    print(f"Pilot cumulative bytes: {used} / {args.max_total_bytes}")


if __name__ == "__main__":
    main()
