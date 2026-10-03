"""Run the five aligned multimodal encoder families and refresh the caption viewer."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = "runs/encoder_expansion_pilot"


def main() -> int:
    commands = [
        ["-m", "caption_bench", "run", "--config", "configs/encoder_expansion_pilot.yaml"],
        [
            "scripts/build_caption_viewer.py",
            "--generated-variants", "variants/gemini_pilot_v2/variants.jsonl",
            "--gemini-run-dir", RUN_DIR,
            "--output", "caption_inceleme.html",
        ],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        if result.returncode:
            return result.returncode
    print(f"Open {ROOT / 'caption_inceleme.html'} and select the expanded encoder experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
