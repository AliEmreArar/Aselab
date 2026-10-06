import numpy as np
import pandas as pd
import pytest
from scripts.build_full150_presentation import paired_interval, validate_coverage, PROFILES


def test_paired_interval_keeps_zero_and_direction():
    assert paired_interval([0, 0, 0]) == pytest.approx([0, 0])
    assert paired_interval([1, 1, 1]) == pytest.approx([1, 1])
    assert paired_interval([-1, -1, -1]) == pytest.approx([-1, -1])


def test_coverage_rejects_missing_model():
    frame = pd.DataFrame({'model': ['one']})
    with pytest.raises(AssertionError):
        validate_coverage(frame, frame, {'models': ['one', 'two']})


def test_merge_preserves_all_image_encoders(tmp_path, monkeypatch):
    import json
    from scripts import merge_encoder_runs as merge
    for name in ['one', 'two']:
        folder = tmp_path / name
        (folder / 'embeddings').mkdir(parents=True)
        (folder / 'variants.jsonl').write_text('{}\n')
        (folder / 'manifest.json').write_text(json.dumps({
            'name': name, 'models': [name], 'notes': [], 'config': {'models': [{'name': name}]}}))
        for file in ['details.csv', 'summary.csv', 'embedding_diagnostics.csv',
                     'image_retrieval_details.csv', 'image_retrieval_summary.csv']:
            pd.DataFrame({'model': [name]}).to_csv(folder / file, index=False)
        np.savez(folder / f'embeddings/{name}.npz', example=[1])
    monkeypatch.setattr(merge, '_agreement', lambda *a: pd.DataFrame({'agreement': [1]}))
    monkeypatch.setattr(merge, 'build_report', lambda *a: None)
    merge.merge(tmp_path / 'one', [tmp_path / 'two'], tmp_path / 'combined')
    for file in ['image_retrieval_details.csv', 'image_retrieval_summary.csv']:
        assert set(pd.read_csv(tmp_path / 'combined' / file).model) == {'one', 'two'}
