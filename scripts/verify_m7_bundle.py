"""Verify M7 bundle integrity, M6 linkage and every scenario summary."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re

from flood_access.analysis_bundle import digest, load_bundle, read_json, read_gzip_rows


def verify(run_id: str, *, deep_source: bool = True) -> dict:
    if re.fullmatch(r"M7S?-[0-9a-f]{12}", run_id) is None:
        raise ValueError("Invalid M7 run ID")
    root = Path("runs") / run_id
    about, roads, evidence, scenarios = load_bundle(root)
    if about["mode"] == "real":
        m6 = Path("runs") / about["source_run_id"]
        if digest(m6 / "run_manifest.json") != about["source_manifest_sha256"]:
            raise ValueError("M7 source M6 manifest changed")
        if deep_source:
            from verify_m6_run import verify as verify_m6
            verify_m6(about["source_run_id"])
        source_rows = read_json(m6 / "scenario_results.json")
        if scenarios != {(row["scenario_id"], row["unknown_or_evidence_policy"], row["bridge_policy"]): row
                         for row in source_rows}:
            raise ValueError("M7 scenario rows differ from M6")
        m6_summary = read_json(m6 / "summary.json")
        if about["baseline"] != m6_summary["baseline"] or \
           about["dates"]["hospital_snapshot"] != m6_summary["hospital_snapshot_date"]:
            raise ValueError("M7 source summary differs")
        from flood_access.case_scenarios import closed_segments
        segment_contract = {key: {"length_m": row[6], "bridge": row[7], "tunnel": row[8]}
                            for key, row in roads.items()}
        source_evidence = {row["segment_id"]: row
                           for row in read_gzip_rows(m6 / "road_evidence.jsonl.gz")}
        if evidence != source_evidence:
            raise ValueError("M7 road evidence differs from M6")
        for entry in about["closure_files"]:
            key = (entry["scenario_id"], entry["policy"], entry["bridge_policy"])
            closed, reasons = closed_segments(segment_contract, evidence, key[0], policy=key[1],
                                              bridge_policy=key[2],
                                              minimum_exposed_m=about["minimum_exposed_supported_m"])
            with gzip.open(root / entry["path"], "rt", encoding="utf-8") as source:
                saved = json.load(source)
            if saved != sorted(closed) or reasons != scenarios[key]["closure"]:
                raise ValueError(f"M7 closed segment IDs differ from M6: {key}")
    elif about["mode"] != "synthetic":
        raise ValueError("Unknown M7 bundle mode")
    return {"run_id": run_id, "status": "verified", "mode": about["mode"],
            "roads": len(roads), "evidence_segments": len(evidence),
            "scenario_rows": len(scenarios), "origins": len(about["origins"]),
            "manifest_sha256": digest(root / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--no-deep-source", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id, deep_source=not args.no_deep_source), indent=2))


if __name__ == "__main__":
    main()
