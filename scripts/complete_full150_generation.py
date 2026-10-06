"""Resume Gemini generation after temporary service errors with bounded restarts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text())
    output = (config.parent / cfg['output_dir']).resolve()
    checkpoint = output / 'checkpoint.json'
    def count():
        return len(json.loads(checkpoint.read_text())['captions']) if checkpoint.exists() else 0
    failures_without_progress = 0
    while True:
        before = count()
        result = subprocess.run([sys.executable, '-u', '-m', 'caption_bench', 'generate',
                                 '--config', str(config)], cwd=ROOT)
        if result.returncode == 0:
            return
        after = count()
        files = sorted((output / 'diagnostics/api_requests').glob('*.json'))
        last = json.loads(files[-1].read_text()) if files else {}
        transient = last.get('http_status') in [500, 502, 503, 504] or last.get('status') == 'connection_error'
        if last.get('http_status') == 429:
            # Only minute-rate exhaustion is resumable; daily/billing quotas stop.
            transient = 'GenerateRequestsPerMinute' in last.get('error_response', '')
        if not transient:
            raise SystemExit('Generation stopped; inspect diagnostics before retrying non-transient failure.')
        failures_without_progress = failures_without_progress + 1 if after == before else 0
        if failures_without_progress >= 3:
            raise SystemExit('Service unavailable for three consecutive restarts without progress; saved outputs preserved.')
        print(f'Temporary service error; saved {after}/150. New request counters on restart after 45 seconds.', flush=True)
        time.sleep(45)


if __name__ == '__main__':
    main()
