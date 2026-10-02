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
    generate_parser = subparsers.add_parser("generate", help="Generate grounded LLM variants using Gemini")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--limit", type=int)
    generate_parser.add_argument("--per-domain", type=int, help="Select a deterministic split-balanced pilot from each domain")
    generate_parser.add_argument("--sample", help="Generate one filename or sample ID")
    generate_parser.add_argument("--dry-run", action="store_true", help="Write a request preview without API calls")
    generate_parser.add_argument("--reuse-response", help="Import a saved model response for --sample without API calls")
    args = parser.parse_args()
    if args.command == "run":
        print(run_experiment(args.config))
    elif args.command == "report":
        print(build_report(args.run_dir))
    elif args.command == "generate":
        from .gemini import generate_variants
        try:
            generate_variants(args.config, limit=args.limit, per_domain=args.per_domain, sample=args.sample,
                              dry_run=args.dry_run, reuse_response=args.reuse_response)
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f"{exc}\n")
    else:
        print("\n".join(available_transforms()))

