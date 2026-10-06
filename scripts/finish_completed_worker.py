"""Release a finished Windows worker stuck in Transformers conversion-thread shutdown."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import psutil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs/full150' / args.model
    config = (ROOT / 'configs/full150' / (args.model + '.yaml')).resolve()
    assert run.joinpath('execution.log').read_text(encoding='utf-8').splitlines()[-1] == str(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['models'] == [args.model] and manifest['caption_count'] == 150
    details = pd.read_csv(run / 'details.csv')
    assert len(details) == manifest['variant_count'] == 1279
    assert details[['sample_id', 'variant_type']].duplicated().sum() == 0
    assert len(pd.read_csv(run / 'discrimination_details.csv')) == 1129
    with np.load(run / 'embeddings' / (args.model + '.npz'), allow_pickle=False) as cache:
        assert len(cache['originals']) == 150 and len(cache['variants']) == 1279
        assert np.isfinite(cache['originals']).all() and np.isfinite(cache['variants']).all()
    candidates = []
    for process in psutil.process_iter(['name', 'cmdline']):
        argv = process.info['cmdline'] or []
        if not (process.info['name'] or '').lower().startswith('python') or '--worker' not in argv:
            continue
        index = argv.index('--worker')
        if index + 1 < len(argv) and Path(argv[index+1]).resolve() == config:
            candidates.append(process)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    released = []
    for process in sorted(candidates, key=lambda p: p.memory_info().rss, reverse=True):
        handle = kernel.OpenProcess(1, False, process.pid)
        if handle:
            try:
                if kernel.TerminateProcess(handle, 0):
                    released.append(process.pid)
            finally:
                kernel.CloseHandle(handle)
    (run / 'shutdown_cleanup.json').write_text(json.dumps(dict(
        reason='All calculations and final log verified; non-daemon Transformers auto_conversion thread blocked shutdown',
        released_worker_pids=released), indent=2), encoding='utf-8')
    print('Verified completed calculation; released shutdown wait:', released)

if __name__ == '__main__':
    main()
