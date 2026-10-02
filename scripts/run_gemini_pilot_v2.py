"""Generate a balanced Gemini v2 pilot, score truncation vs chunking, and refresh the viewer."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = "variants/gemini_pilot_v2/variants.jsonl"
RUN_DIR = "runs/gemini_pilot_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-domain", type=int, default=5, help="Split-balanced examples per domain")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generation = [
        "-m", "caption_bench", "generate", "--config", "configs/gemini_pilot_v2.yaml",
        "--per-domain", str(args.per_domain),
    ]
    if args.dry_run:
        generation.append("--dry-run")
    commands = [generation]
    viewer = [
        "scripts/build_caption_viewer.py", "--generated-variants", GENERATED,
        "--gemini-run-dir", RUN_DIR, "--output", "caption_inceleme.html",
    ]
    if not args.dry_run:
        commands.append(viewer)
        if not args.generate_only:
            commands.extend([
                ["-m", "caption_bench", "run", "--config", "configs/gemini_pilot_v2_benchmark.yaml"],
                viewer,
            ])
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        if result.returncode:
            print("Pilot step failed; completed caption batches remain resumable.", file=sys.stderr)
            return result.returncode
    if not args.dry_run:
        print(f"Open {ROOT / 'caption_inceleme.html'} and select the Gemini experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

