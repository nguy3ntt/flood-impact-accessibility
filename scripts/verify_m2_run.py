"""Verify all source and derived hashes for a completed local M2 pilot run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(run_id: str) -> dict:
    root = Path("data/processed/m2_runs") / run_id
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or manifest["status"] != "complete" or status["status"] != "complete":
        raise ValueError("Run is not complete or run IDs differ")
    if status["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Run manifest hash differs from recorded status")
    checked_outputs = 0
    for name, expected in manifest["output_sha256"].items():
        path = Path(name)
        if ".." in path.parts or (not path.is_relative_to(root) and not path.is_relative_to(Path("reports/m2") / run_id)):
            raise ValueError(f"Output path is outside this run: {path}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed output: {path}")
        checked_outputs += 1
    actual = {path.as_posix() for directory in (root, Path("reports/m2") / run_id)
              for path in directory.rglob("*") if path.is_file() and path != manifest_path}
    if actual != set(manifest["output_sha256"]):
        raise ValueError("Output file set differs from the run manifest")
    for source_id, record in manifest["source_objects"].items():
        path = Path(record["local_path"])
        if ".." in path.parts or not path.is_relative_to("data/raw") or not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Missing or changed source: {source_id}")
    return {"run_id": run_id, "status": "verified", "outputs": checked_outputs,
            "sources": len(manifest["source_objects"]), "manifest_sha256": digest(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="M2 run ID printed by build_m2_pilot.py")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))


if __name__ == "__main__":
    main()
