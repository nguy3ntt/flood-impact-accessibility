"""Build an immutable local M7 analyst bundle from verified M6 or synthetic fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_access.analysis_bundle import build_real, build_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m6-run-id", default="M6-7b030cf0d570")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        result = build_synthetic()
    else:
        from verify_m6_run import verify as verify_m6
        verify_m6(args.m6_run_id)
        result = build_real(Path("runs") / args.m6_run_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
