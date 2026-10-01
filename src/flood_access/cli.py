import argparse
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
    return {
        "status": "m2_pilot" if m2_run_id else ("m1_pilot" if present else "scaffold"),
        "local_pilot_files_present": present,
        "m2_pilot_run_id": m2_run_id,
        "models_trained": False,
        "application_implemented": False,
        "next_step": "M3 mapping baselines" if m2_run_id else ("M2 canonical data pipeline" if present else "M1 pilot acquisition"),
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
