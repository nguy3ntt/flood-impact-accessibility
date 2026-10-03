import argparse
import hashlib
import json
from pathlib import Path
from .network import synthetic_demo


def local_status(root: Path) -> dict:
    base = root / "data/raw/sen1floods11/v1.1"
    pilot = [base / "Sen1Floods11_Metadata.geojson"]
    for chip in ("Bolivia_103757", "Ghana_103272", "Spain_7370579"):
        for layer in ("LabelHand", "S1Hand", "S2Hand"):
            pilot.append(
                base / "data/flood_events/HandLabeled" / layer / f"{chip}_{layer}.tif"
            )
    present = all(path.is_file() for path in pilot)
    completed_runs = []
    for manifest_path in (root / "data/processed/m2_runs").glob("M2-*/run_manifest.json"):
        run_id = manifest_path.parent.name
        status_path = root / "runs" / run_id / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete" and status.get("run_id") == run_id:
                completed_runs.append((status.get("completed_at_utc", ""), run_id))
        except (OSError, ValueError):
            continue
    m2_run_id = max(completed_runs)[1] if present and completed_runs else None
    completed_m3 = []
    for status_path in (root / "runs").glob("M3-*/status.json"):
        run_id = status_path.parent.name
        manifest_path = status_path.parent / "run_manifest.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if (status.get("status") == "complete" and status.get("run_id") == run_id
                    and status.get("manifest_sha256") == digest
                    and manifest.get("status") == "complete" and manifest.get("run_id") == run_id):
                completed_m3.append((status.get("completed_at_utc", ""), run_id))
        except (OSError, ValueError):
            continue
    m3_run_id = max(completed_m3)[1] if completed_m3 else None
    def completed_mapping_run(prefix: str) -> str | None:
        completed = []
        for status_path in (root / "runs").glob(f"{prefix}-*/status.json"):
            run_id = status_path.parent.name
            manifest_path = status_path.parent / "run_manifest.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if (status.get("status") == "complete" and status.get("run_id") == run_id
                        and status.get("manifest_sha256") == digest
                        and manifest.get("status") == "complete" and manifest.get("run_id") == run_id
                        and manifest.get("m3_reference_run_id") == m3_run_id):
                    completed.append((status.get("completed_at_utc", ""), run_id))
            except (OSError, ValueError):
                continue
        return max(completed)[1] if m3_run_id and completed else None
    m4_run_id = completed_mapping_run("M4")
    m4_budget_run_id = completed_mapping_run("M4B")
    m5_run_id = completed_mapping_run("M5")
    if m5_run_id:
        m5_manifest = json.loads((root / "runs" / m5_run_id / "run_manifest.json").read_text(encoding="utf-8"))
        if m5_manifest.get("m4_control_run_id") != m4_budget_run_id:
            m5_run_id = None
    completed_m6 = []
    if m5_run_id and m2_run_id:
        for status_path in (root / "runs").glob("M6-*/status.json"):
            run_id = status_path.parent.name
            manifest_path = status_path.parent / "run_manifest.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if (status.get("status") == "complete" and status.get("run_id") == run_id
                        and status.get("manifest_sha256") == digest
                        and manifest.get("status") == "complete" and manifest.get("run_id") == run_id
                        and manifest.get("m2_run_id") == m2_run_id
                        and manifest.get("m5_run_id") == m5_run_id):
                    completed_m6.append((status.get("completed_at_utc", ""), run_id))
            except (OSError, ValueError):
                continue
    m6_run_id = max(completed_m6)[1] if completed_m6 else None
    completed_m7 = []
    completed_m7_synthetic = []
    for prefix, destination in (("M7", completed_m7), ("M7S", completed_m7_synthetic)):
        for status_path in (root / "runs").glob(f"{prefix}-*/status.json"):
            run_id = status_path.parent.name
            manifest_path = status_path.parent / "manifest.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                source_ok = (manifest.get("source_run_id") == m6_run_id if prefix == "M7"
                             else manifest.get("source_run_id") is None)
                if (status.get("status") == "complete" and status.get("run_id") == run_id
                        and status.get("manifest_sha256") == digest
                        and manifest.get("status") == "complete" and manifest.get("run_id") == run_id
                        and source_ok):
                    destination.append((status.get("completed_at_utc", ""), run_id))
            except (OSError, ValueError):
                continue
    m7_run_id = max(completed_m7)[1] if completed_m7 else None
    m7_synthetic_run_id = max(completed_m7_synthetic)[1] if completed_m7_synthetic else None
    completed_m8 = []
    if m7_run_id and m5_run_id:
        for status_path in (root / "runs").glob("M8-*/status.json"):
            run_id = status_path.parent.name
            manifest_path = status_path.parent / "run_manifest.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if (status.get("status") == "complete" and status.get("run_id") == run_id
                        and status.get("manifest_sha256") == digest
                        and manifest.get("schema_version") == "m8_final_run_v1"
                        and manifest.get("status") == "complete" and manifest.get("run_id") == run_id
                        and manifest.get("events") == ["nigeria_20180921", "somalia_20180507"]
                        and manifest.get("chips") == 44):
                    completed_m8.append((status.get("completed_at_utc", ""), run_id))
            except (OSError, ValueError):
                continue
    m8_run_id = max(completed_m8)[1] if completed_m8 else None
    return {
        "status": ("m8_final_research" if m8_run_id else
                   "m7_analyst_application" if m7_run_id else
                   "m7_synthetic_demo" if m7_synthetic_run_id and not m6_run_id else
                   "m6_accessibility_scenarios" if m6_run_id else
                   "m5_evidence" if m5_run_id and m4_run_id and m4_budget_run_id else
                   "m4_development" if m4_run_id and m4_budget_run_id else
                   "m3_development" if m3_run_id else
                   "m2_pilot" if m2_run_id else "m1_pilot" if present else "scaffold"),
        "local_pilot_files_present": present,
        "m2_pilot_run_id": m2_run_id,
        "m3_run_id": m3_run_id,
        "m4_fusion_run_id": m4_run_id,
        "m4_budget_run_id": m4_budget_run_id,
        "m5_run_id": m5_run_id,
        "m6_run_id": m6_run_id,
        "m7_run_id": m7_run_id,
        "m7_synthetic_run_id": m7_synthetic_run_id,
        "m8_run_id": m8_run_id,
        "models_trained": bool(m3_run_id),
        "application_implemented": bool(m7_run_id or m7_synthetic_run_id),
        "next_step": ("Owner review and manual commit; licence and real-data redistribution gates remain"
                      if m8_run_id else
                      "M8 final research and release audit" if m7_run_id else
                      "Synthetic analyst demo ready; real-data rebuild needs source acquisition"
                      if m7_synthetic_run_id and not m6_run_id else
                      "M7 analyst application and reproducible serving" if m6_run_id else
                      "M6 road exposure and hospital accessibility" if m5_run_id and m4_run_id and m4_budget_run_id else
                      "M5 calibration and coherent uncertainty" if m4_run_id and m4_budget_run_id else
                      "M4 fusion and transfer" if m3_run_id else
                      "M3 mapping baselines" if m2_run_id else
                      "M2 canonical data pipeline" if present else "M1 pilot acquisition"),
    }

def main():
    parser = argparse.ArgumentParser(description="Flood accessibility research scaffold")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    demo = commands.add_parser("demo", help="Run an explicitly synthetic graph scenario")
    demo.add_argument("--scenario", choices=["baseline", "exposed_only", "conservative"],
                      default="conservative")
    args = parser.parse_args()
    result = (local_status(Path.cwd())
              if args.command == "status" else synthetic_demo(args.scenario))
    print(json.dumps(result, indent=2, allow_nan=False))
