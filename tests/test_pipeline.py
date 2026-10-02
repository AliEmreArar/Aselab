from pathlib import Path

import numpy as np
import pandas as pd

from caption_bench.data import Caption
from caption_bench.encoders import EncodingBatch, HashingEncoder
from caption_bench.metrics import score_image_retrieval, score_variants, summarize, summarize_image_retrieval
from caption_bench.transforms import generate_variants


def test_identity_retrieves_itself_first():
    captions = [
        Caption("person::train::1.jpg", "person", "train", "1.jpg", "red coat white shoes"),
        Caption("person::train::2.jpg", "person", "train", "2.jpg", "blue shirt black boots"),
    ]
    variants = generate_variants(captions, [{"name": "identity"}], seed=3)
    encoder = HashingEncoder({"name": "hash", "backend": "hash", "dimension": 128})
    originals = encoder.encode([caption.text for caption in captions])
    changed = encoder.encode([variant.text for variant in variants])
    details = score_variants("hash", captions, variants, originals, changed)
    assert details["recall_at_1"].tolist() == [1, 1]
    assert details["cosine"].min() > 0.999


def test_summary_has_expected_columns():
    frame = pd.DataFrame(
        [{"model": "m", "domain": "d", "variant_type": "v", "sample_id": "1", "cosine": .8, "margin": .2, "reciprocal_rank": 1., "recall_at_1": 1, "recall_at_5": 1, "original_truncated": False, "variant_truncated": False, "original_tokens": 10, "variant_tokens": 5}]
    )
    result = summarize(frame)
    assert result.loc[0, "recall_at_1"] == 1
    assert result.loc[0, "cosine_mean"] == .8


def test_transforms_are_deterministic():
    caption = Caption("x", "face", "train", "x.jpg", "Adult, oval face, dark hair, rounded chin, visible eyes.")
    specs = [{"name": "synonym", "probability": 1.0}, {"name": "clause_exchange"}]
    assert generate_variants([caption], specs, 42) == generate_variants([caption], specs, 42)


def test_image_retrieval_ranks_the_matching_image():
    captions = [
        Caption("person::train::1.jpg", "person", "train", "1.jpg", "red coat"),
        Caption("person::train::2.jpg", "person", "train", "2.jpg", "blue shirt"),
    ]
    variants = generate_variants(captions, [{"name": "identity"}], seed=3)
    batch = EncodingBatch(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        token_counts=[2, 2],
        truncated=[False, False],
    )
    images = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    details = score_image_retrieval("multimodal", captions, variants, batch, images)
    assert details["image_rank"].tolist() == [1, 1]
    summary = summarize_image_retrieval(details)
    assert summary["image_recall_at_1"].iloc[0] == 1.0
    assert summary["matched_identity_recall_at_1"].iloc[0] == 1.0
    assert summary["delta_image_recall_at_1"].iloc[0] == 0.0

