from __future__ import annotations

import argparse

from .report import build_report
from .runner import run_experiment
from .transforms import available_transforms


def main() -> None:
    parser = argparse.ArgumentParser(prog="caption-bench", description="Compare caption variants across text encoders")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config")
    run_parser.add_argument("--config", required=True)
    report_parser = subparsers.add_parser("report", help="Rebuild the HTML report from a run directory")
    report_parser.add_argument("--run-dir", required=True)
    subparsers.add_parser("list-transforms", help="List built-in caption transformations")
    args = parser.parse_args()
    if args.command == "run":
        print(run_experiment(args.config))
    elif args.command == "report":
        print(build_report(args.run_dir))
    else:
        print("\n".join(available_transforms()))

