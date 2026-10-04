import pytest
import json
import numpy as np
import pandas as pd
from caption_bench.calibration import analyze
from caption_bench.calibration import negative_metrics
from scripts.prepare_controlled_pilot import make_controls


def test_high_cosine_is_not_discrimination():
    result = negative_metrics(.98, [.99, .995, .985])
    assert result["negative_percentile"] == 0
    assert result["hardest_margin"] == pytest.approx(-.015)


def test_ties_get_half_credit():
    result = negative_metrics(.9, [.8, .9, 1.0])
    assert result["negative_percentile"] == .5


def test_face_accessory_control_preserves_wearing_claim():
    controls = make_controls("She is wearing a tiara, veil, and necklace.", "face")
    assert controls[0][2] == "She wears a tiara, veil, and necklace."


def test_end_to_end_same_domain_and_conservative_ties(tmp_path):
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"models": ["test"]}))
    variant = {"sample_id": "face::a", "variant_type": "summary",
               "domain": "face", "relation": "information_reduced"}
    (tmp_path / "variants.jsonl").write_text(json.dumps(variant) + "\n")
    np.savez(tmp_path / "embeddings/test.npz",
             caption_ids=["face::a", "face::b", "vehicle::c"],
             variant_ids=["face::a::summary"],
             originals=[[1., 0.], [1., 0.], [1., 0.]], variants=[[1., 0.]])
    row = analyze(tmp_path).iloc[0]
    assert row["negative_count"] == 1  # Vehicle must not become a face competitor.
    assert row["negative_percentile"] == .5
    assert row["rank"] == 2
    assert row["recall_at_1"] == 0
    assert row["reciprocal_rank"] == .5
    assert len(pd.read_csv(tmp_path / "discrimination_summary.csv")) == 1
