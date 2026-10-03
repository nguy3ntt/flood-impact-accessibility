"""Serve one validated M7 bundle on localhost only."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import uvicorn

from flood_access.serving import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Exact M7-... or M7S-... bundle ID")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if re.fullmatch(r"M7S?-[0-9a-f]{12}", args.run_id) is None or not 1024 <= args.port <= 65535:
        parser.error("Invalid M7 bundle ID or localhost port")
    app = create_app(Path("runs") / args.run_id)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
