"""Download model weights in verified, resumable HTTP ranges on unstable connections."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.model_cache')
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download
import requests
from filelock import FileLock

JOBS = [('openai/clip-vit-large-patch14', None, 'model.safetensors'),
        ('BAAI/AltCLIP', None, 'pytorch_model.bin'),
        ('BAAI/bge-m3', None, 'pytorch_model.bin'),
        ('microsoft/deberta-v3-base', None, 'pytorch_model.bin'),
        ('jinaai/jina-clip-v2', 'e10d47f5691d0454a0fb5d13f46f2199b74cb436', 'model.safetensors'),
        ('timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k', None, 'open_clip_model.safetensors')]
HF_SETUP_LOCK = threading.Lock()


def record(repo, filename, metadata):
    path = ROOT / '.model_cache/full150_downloads.json'
    with FileLock(str(path) + '.lock'):
        data = json.loads(path.read_text()) if path.exists() else {}
        data[repo] = dict(commit=metadata.commit_hash, filename=filename,
                          sha256=metadata.etag, bytes=metadata.size)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temp.replace(path)


def download(repo, revision, filename):
    url = hf_hub_url(repo, filename, revision=revision)
    metadata = get_hf_file_metadata(url)
    # HF's nested tqdm setup is process-global; serialize only this small-file stage.
    with HF_SETUP_LOCK:
        snapshot = Path(snapshot_download(repo, revision=metadata.commit_hash,
                        allow_patterns=['*.json', '*.txt', '*.model', '*.py'], max_workers=2))
    target = snapshot / filename
    if target.exists() and target.stat().st_size == metadata.size:
        with target.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != metadata.etag:
                raise ValueError(f'Cached weight checksum mismatch: {repo}')
        record(repo, filename, metadata)
        print(f'Already downloaded: {repo}', flush=True)
        return
    partial = target.with_suffix(target.suffix + '.partial')
    chunk_size = 16 * 1024 * 1024
    local = threading.local()
    def fetch_range(bounds):
        start, end = bounds
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        for attempt in range(5):
            try:
                response = local.session.get(metadata.location,
                    headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 45))
                response.raise_for_status()
                expected = f'bytes {start}-{end}/{metadata.size}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Server did not return the requested byte range')
                if len(response.content) != end-start+1:
                    raise ValueError('Incomplete HTTP range')
                return response.content
            except (requests.RequestException, ValueError) as exc:
                if attempt == 4:
                    raise
                print(f'{repo}: retry range {start} ({type(exc).__name__})', flush=True)
                time.sleep(3 * (attempt+1))
    with ThreadPoolExecutor(max_workers=4) as pool, partial.open('ab') as out:
        while out.tell() < metadata.size:
            start = out.tell()
            bounds = [(s, min(s+chunk_size, metadata.size)-1)
                      for s in range(start, min(start+4*chunk_size, metadata.size), chunk_size)]
            # map preserves byte order; only complete, validated chunks are appended.
            for content in pool.map(fetch_range, bounds):
                out.write(content)
                out.flush()
            print(f'{repo}: {out.tell()/metadata.size:.1%}', flush=True)
    with partial.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if len(metadata.etag) != 64 or digest != metadata.etag:
        raise ValueError(f'Weight checksum mismatch for {repo}')
    partial.replace(target)
    record(repo, filename, metadata)
    print(f'Verified weight: {repo}', flush=True)


if __name__ == '__main__':
    def job(args):
        try:
            with FileLock(str(ROOT / '.model_cache' / (args[0].replace('/', '--') + '.study.lock'))):
                download(*args)
            return None
        except Exception as exc:
            print(f'Download failed: {args[0]} ({type(exc).__name__}): {str(exc)[:400]}', flush=True)
            return args[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = [name for name in pool.map(job, JOBS) if name]
    if failed:
        raise SystemExit('Failed repositories: ' + ', '.join(failed))
