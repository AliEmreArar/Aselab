"""Verify cached text scores and exact caption coverage without rerunning encoders."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'runs/full150'
RESUMED = ['siglip2_b16_224', 'eva02_clip_l14', 'altclip', 'bge_m3']


def main():
    design = json.loads((STUDY / 'design.json').read_text(encoding='utf-8'))
    generated = [json.loads(line) for line in (ROOT / 'variants/full150/variants.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(r['sample_id'], r['variant_type']): r['text'] for r in generated}
    models, reference = [], None
    for name in design['models']:
        folder = STUDY / name
        assert (folder / 'completed.json').exists(), f'Incomplete {name}'
        rows = [json.loads(line) for line in (folder / 'variants.jsonl').read_text(encoding='utf-8').splitlines()]
        if reference is None:
            reference = rows
        assert rows == reference, f'Different captions for {name}'
        assert all(expected[r['sample_id'], r['variant_type']] == r['text']
                   for r in rows if r['variant_type'] != 'identity')
        with np.load(folder / 'embeddings' / f'{name}.npz', allow_pickle=False) as z:
            assert np.isfinite(z['originals']).all() and np.isfinite(z['variants']).all()
            oi = {str(s): i for i, s in enumerate(z['caption_ids'])}
            vi = {str(s): i for i, s in enumerate(z['variant_ids'])}
            assert len(oi) == 150 and len(vi) == 1279
            original, variants = z['originals'].astype(float), z['variants'].astype(float)
        original /= np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-12)
        variants /= np.maximum(np.linalg.norm(variants, axis=1, keepdims=True), 1e-12)
        details = pd.read_csv(folder / 'details.csv')
        assert len(details) == 1279 and not details.duplicated(['sample_id', 'variant_type']).any()
        actual = np.array([original[oi[r.sample_id]] @ variants[vi[r.sample_id+'::'+r.variant_type]]
                           for r in details.itertuples()])
        assert np.allclose(actual, details.cosine, atol=2e-5), f'Cosine cache mismatch {name}'
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        spec = manifest['config']['models'][0]
        assert spec['device'] == 'cpu' and spec['dtype'] == 'float32'
        if spec.get('image_retrieval'):
            images = pd.read_csv(folder / 'image_retrieval_details.csv')
            assert len(images) == 1279 and not images.duplicated(['sample_id', 'variant_type']).any()
            assert (images.image_pool_size == 50).all()
            assert np.array_equal(images.image_recall_at_1, (images.image_rank == 1).astype(int))
            assert np.array_equal(images.image_recall_at_5, (images.image_rank <= 5).astype(int))
        record = dict(model=name, caption_rows=len(rows), text_rows=len(details),
                      image_rows=1279 if spec.get('image_retrieval') else 0,
                      max_cosine_error=float(np.max(np.abs(actual-details.cosine))),
                      device=spec['device'], dtype=spec['dtype'], revision=spec.get('revision'))
        if name in RESUMED:
            repo = spec['model_id'] if spec['backend'] != 'open_clip' else 'timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k'
            cache = Path.home() / '.cache/huggingface/hub' / ('models--'+repo.replace('/', '--'))
            ref = cache / 'refs/main'
            record['local_cached_main_commit'] = ref.read_text().strip() if ref.exists() else None
        models.append(record)
    data = dict(models=models, validation='exact variants, cached text cosine, coverage, image R@K from ranks',
                image_vectors_not_cached='Image cosine cannot be independently recomputed from the available npz files',
                original_host_versions=design['package_versions'],
                resumed_models=RESUMED, resumed_platform=platform.platform(),
                resumed_package_versions={p: importlib.metadata.version(p) for p in design['package_versions']},
                variants_sha256=hashlib.sha256((ROOT / 'variants/full150/variants.jsonl').read_bytes()).hexdigest())
    (STUDY / 'completion_validation.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'Validated {len(models)} models, identical 1279 caption rows each; 6 image retrieval runs.')


if __name__ == '__main__':
    main()
