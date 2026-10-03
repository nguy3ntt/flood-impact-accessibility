"""Copy exactly the current Git candidates into a new ignored clean-room directory."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / "tmp" / f"m8_clean_copy_{stamp}"
    if target.exists():
        raise FileExistsError(target)
    candidates = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).split(b"\0")
    copied = []
    for encoded in candidates:
        if not encoded:
            continue
        relative = Path(encoded.decode("utf-8"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe Git candidate path: {relative}")
        source = root / relative
        if not source.is_file():
            continue  # Deleted tracked files are not candidates in the working tree.
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())
    raw_files = [path for path in (target / "data/raw").rglob("*")
                 if path.is_file() and path.name != ".gitkeep"]
    if not copied or (target / ".project").exists() or raw_files:
        raise ValueError("Clean copy is empty or contains local-only material")
    print(json.dumps({"copy": str(target), "candidate_files": len(copied),
                      "contains_internal_notes": False, "contains_raw_data": False}, indent=2))


if __name__ == "__main__":
    main()
