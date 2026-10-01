import argparse
import json
from .network import synthetic_demo

def main():
    parser = argparse.ArgumentParser(description="Flood accessibility research scaffold")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    demo = commands.add_parser("demo", help="Run an explicitly synthetic graph scenario")
    demo.add_argument("--scenario", choices=["baseline", "exposed_only", "conservative"],
                      default="conservative")
    args = parser.parse_args()
    result = ({"status": "scaffold", "real_data_acquired": False,
               "models_trained": False, "application_implemented": False,
               "next_step": "M0 environment and M1 feasibility audit"}
              if args.command == "status" else synthetic_demo(args.scenario))
    print(json.dumps(result, indent=2, allow_nan=False))
