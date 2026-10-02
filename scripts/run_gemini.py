"""Generate Gemini captions, optionally score encoders, and refresh the viewer."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", help="One filename or sample ID")
    parser.add_argument("--limit", type=int, help="Limit the selected captions")
    parser.add_argument("--generate-only", action="store_true", help="Generate and show captions without running encoders")
    parser.add_argument("--dry-run", action="store_true", help="Preview the request without any API calls")
    parser.add_argument("--reuse-response", help="Import a saved model answer instead of making API requests")
    args = parser.parse_args()
    # Reuse encoder weights downloaded by the local benchmark workflow.
    os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache"))
    generation = ["-m", "caption_bench", "generate", "--config", "configs/gemini_variants.yaml"]
    if args.sample:
        generation += ["--sample", args.sample]
    if args.limit is not None:
        generation += ["--limit", str(args.limit)]
    if args.dry_run:
        generation += ["--dry-run"]
    if args.reuse_response:
        generation += ["--reuse-response", args.reuse_response]
    commands = [generation]
    if not args.dry_run:
        commands.append(["scripts/build_caption_viewer.py"])
        if not args.generate_only:
            commands.extend([["-m", "caption_bench", "run", "--config", "configs/gemini_benchmark.yaml"],
                             ["scripts/build_caption_viewer.py"]])
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        if result.returncode:
            print("Step failed. Check the error and variants/gemini/diagnostics before retrying. Only completed caption batches are exported; existing completed batches are preserved.", file=sys.stderr)
            return result.returncode
    if not args.dry_run:
        print(f"Open {ROOT / 'caption_inceleme.html'} and select the Gemini experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
