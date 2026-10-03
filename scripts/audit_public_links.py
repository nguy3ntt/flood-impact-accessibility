"""Check local Markdown links in Git-candidate documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")


def main() -> None:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    missing = []
    count = 0
    for source in files:
        for match in LINK.finditer(source.read_text(encoding="utf-8")):
            raw = match.group(1).split()[0].strip("<>")
            if raw.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = (source.parent / unquote(raw.split("#", 1)[0])).resolve()
            count += 1
            if not target.is_relative_to(ROOT) or not target.exists():
                missing.append(f"{source.relative_to(ROOT)} -> {raw}")
    if missing:
        raise ValueError("Missing local links:\n" + "\n".join(missing))
    print(f"Checked {count} local links across {len(files)} Markdown files; all targets exist.")


if __name__ == "__main__":
    main()
