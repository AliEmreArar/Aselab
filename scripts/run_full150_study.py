"""Full coverage study; each encoder is isolated and completed runs are resumable."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from filelock import FileLock
from caption_bench.data import load_captions, read_jsonl, write_jsonl

PROFILES = ['llm_brief', 'llm_balanced', 'llm_detailed', 'llm_human_description',
            'llm_attribute_list', 'llm_distinctive_first']
STUDY = ROOT / 'runs/full150'


def prepare():
    from scripts.prepare_controlled_pilot import make_controls
    captions = load_captions(ROOT / 'caption_samples_50/all_samples.jsonl')
    rows = sum((read_jsonl(ROOT / f'variants/{name}/variants.jsonl')
                for name in ['full150_standard', 'full150_alternative']), [])
    expected = {(c.sample_id, p) for c in captions for p in PROFILES}
    actual = [(r['sample_id'], r['variant_type']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f'Generation incomplete: {len(set(actual) & expected)}/{len(expected)} required variants')
    originals = {c.sample_id: c.text for c in captions}
    controls = []
    for row in rows:
        if row['source_sha256'] != hashlib.sha256(originals[row['sample_id']].encode()).hexdigest():
            raise ValueError(f'Source mismatch: {row["sample_id"]}')
        if row['variant_type'] == 'llm_balanced':
            for label, relation, text, note in make_controls(row['text'], row['sample_id'].split('::')[0]):
                controls.append(dict(sample_id=row['sample_id'], variant_type=label, text=text,
                                     relation=relation, generator='controlled:single-span', change_note=note,
                                     control_base='llm_balanced', source_sha256=row['source_sha256']))
    write_jsonl(ROOT / 'variants/full150/variants.jsonl', rows + controls)
    config = yaml.safe_load((ROOT / 'configs/encoder_expansion_pilot.yaml').read_text())
    config['models'].insert(0, dict(name='clip_b16', backend='hf_text_features',
        model_id='openai/clip-vit-base-patch16', revision='refs/pr/16', use_safetensors=True,
        max_length=77, batch_size=8, image_retrieval=True))
    config['variant_files'] = ['../variants/full150/variants.jsonl']
    configs = ROOT / 'configs/full150'
    configs.mkdir(exist_ok=True)
    downloads_path = ROOT / '.model_cache/full150_downloads.json'
    downloads = json.loads(downloads_path.read_text()) if downloads_path.exists() else {}
    for model in config['models']:
        if model['model_id'] in downloads:
            model['revision'] = downloads[model['model_id']]['commit']
        elif model['backend'] != 'open_clip' and not model.get('revision'):
            snapshots = ROOT / '.model_cache/hub' / ('models--' + model['model_id'].replace('/', '--')) / 'snapshots'
            ready = [p for p in snapshots.glob('*')
                     if (p / 'model.safetensors').exists() or (p / 'pytorch_model.bin').exists()]
            if len(ready) == 1:
                model['revision'] = ready[0].name
        model['device'] = 'cpu'
        model['dtype'] = 'float32'
        model['batch_size'] = min(model.get('batch_size', 8), 8)
        if model['backend'] == 'open_clip':
            model['precision'] = 'fp32'
        child = dict(config, name=f'full150-{model["name"]}', models=[model],
                     dataset='../../caption_samples_50/all_samples.jsonl',
                     variant_files=['../../variants/full150/variants.jsonl'],
                     output_dir=f'../../runs/full150/{model["name"]}')
        (configs / f'{model["name"]}.yaml').write_text(yaml.safe_dump(child, sort_keys=False), encoding='utf-8')
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / 'design.json').write_text(json.dumps(dict(caption_count=len(captions),
        profiles=PROFILES, models=[m['name'] for m in config['models']],
        source='Existing source captions; Gemini does not view images',
        excluded={'mamba3': 'CUDA/Triton unavailable; raw causal LM is not image aligned'},
        pilot_reuse=15, controls=len(controls), precision='CPU float32',
        package_versions={p: importlib.metadata.version(p) for p in
                          ['torch', 'transformers', 'sentence-transformers', 'open-clip-torch', 'numpy']},
        query_selection='25 development / 25 held-out per domain; pilot examples assigned to development',
        gallery='All 50 images/captions of the same domain; exact source-file retrieval'), indent=2), encoding='utf-8')
    print(f'Prepared {len(captions)} sources, {len(rows)} Gemini variants, {len(controls)} controls', flush=True)


def signature(config):
    return hashlib.sha256(config.read_bytes() + (ROOT / 'variants/full150/variants.jsonl').read_bytes()
                          + (ROOT / 'caption_samples_50/all_samples.jsonl').read_bytes()
                          + (ROOT / 'caption_bench/encoders.py').read_bytes()
                          + (ROOT / 'caption_bench/metrics.py').read_bytes()).hexdigest()


def run(models):
    failures = []
    order = {name: i for i, name in enumerate(models or [])}
    configs = sorted((ROOT / 'configs/full150').glob('*.yaml'),
                     key=lambda p: (order.get(p.stem, len(order)), p.stem))
    for config in configs:
        name = config.stem
        if models and name not in models:
            continue
        out = STUDY / name
        out.mkdir(parents=True, exist_ok=True)
        with FileLock(str(out / '.execution.lock')):
            complete = out / 'completed.json'
            sig = signature(config)
            if complete.exists() and json.loads(complete.read_text())['signature'] == sig:
                print(f'Skip completed: {name}', flush=True)
                continue
            print(f'Starting encoder: {name}; log: {out / "execution.log"}', flush=True)
            with (out / 'execution.log').open('w', encoding='utf-8') as log:
                proc = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                       '--worker', str(config)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                failures.append(name)
                print(f'FAILED {name}; inspect execution.log', flush=True)
            else:
                complete.write_text(json.dumps({'signature': sig}), encoding='utf-8')
                print(f'Completed: {name}', flush=True)
    if failures:
        raise RuntimeError(f'Failed encoders: {failures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        # A resumed run may use an already populated cache on a different host.
        os.environ.setdefault('HF_HOME', str(ROOT / '.model_cache'))
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['DISABLE_SAFETENSORS_CONVERSION'] = 'true'
        import torch
        torch.set_num_threads(6)
        from caption_bench import runner
        original_factory = runner.create_encoder
        def logged_encoder(spec):
            encoder = original_factory(spec)
            for method_name in ('encode', 'encode_images'):
                method = getattr(encoder, method_name)
                def logged(values, _method=method, _name=method_name):
                    print(f'{encoder.name}: {_name} started ({len(values)} items)', flush=True)
                    result = _method(values)
                    print(f'{encoder.name}: {_name} finished', flush=True)
                    return result
                setattr(encoder, method_name, logged)
            return encoder
        runner.create_encoder = logged_encoder
        from caption_bench.calibration import analyze
        output = runner.run_experiment(args.worker)
        analyze(output)
        print(output, flush=True)
    else:
        if args.prepare:
            prepare()
        if args.run:
            run(args.models)
