from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Caption, Variant
from .encoders import EncodingBatch


def score_variants(
    model_name: str,
    captions: list[Caption],
    variants: list[Variant],
    original_batch: EncodingBatch,
    variant_batch: EncodingBatch,
) -> pd.DataFrame:
    caption_index = {caption.sample_id: idx for idx, caption in enumerate(captions)}
    domains: dict[str, list[int]] = {}
    for idx, caption in enumerate(captions):
        domains.setdefault(caption.domain, []).append(idx)

    rows: list[dict] = []
    for variant_idx, variant in enumerate(variants):
        target_idx = caption_index[variant.sample_id]
        caption = captions[target_idx]
        candidate_indices = domains[caption.domain]
        scores = variant_batch.embeddings[variant_idx] @ original_batch.embeddings[candidate_indices].T
        target_pos = candidate_indices.index(target_idx)
        target_score = float(scores[target_pos])
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == target_pos)[0][0]) + 1
        negative_scores = np.delete(scores, target_pos)
        hardest_negative = float(np.max(negative_scores)) if len(negative_scores) else float("nan")
        rows.append(
            {
                "model": model_name,
                "sample_id": caption.sample_id,
                "domain": caption.domain,
                "split": caption.split,
                "filename": caption.filename,
                "variant_type": variant.variant_type,
                "generator": variant.generator,
                "relation": variant.relation,
                "original_text": caption.text,
                "variant_text": variant.text,
                "original_words": len(caption.text.split()),
                "variant_words": len(variant.text.split()),
                "original_tokens": original_batch.token_counts[target_idx],
                "variant_tokens": variant_batch.token_counts[variant_idx],
                "original_truncated": original_batch.truncated[target_idx],
                "variant_truncated": variant_batch.truncated[variant_idx],
                "cosine": target_score,
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "recall_at_1": int(rank <= 1),
                "recall_at_5": int(rank <= 5),
                "hardest_negative_cosine": hardest_negative,
                "margin": target_score - hardest_negative,
                "retrieval_pool_size": len(candidate_indices),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean_interval(values: pd.Series, seed: int, iterations: int = 2_000) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model", "domain", "variant_type"]
    summary = (
        details.groupby(group_columns, dropna=False)
        .agg(
            n=("sample_id", "size"),
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            cosine_p10=("cosine", lambda x: x.quantile(0.10)),
            cosine_median=("cosine", "median"),
            margin_mean=("margin", "mean"),
            mrr=("reciprocal_rank", "mean"),
            recall_at_1=("recall_at_1", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            original_truncation_rate=("original_truncated", "mean"),
            variant_truncation_rate=("variant_truncated", "mean"),
            original_tokens_mean=("original_tokens", "mean"),
            variant_tokens_mean=("variant_tokens", "mean"),
        )
        .reset_index()
    )
    ci_rows: list[dict] = []
    for group_idx, (key, group) in enumerate(details.groupby(group_columns, sort=True, dropna=False)):
        cosine_low, cosine_high = _bootstrap_mean_interval(group["cosine"], 41_000 + group_idx)
        r1_low, r1_high = _bootstrap_mean_interval(group["recall_at_1"], 73_000 + group_idx)
        ci_rows.append(
            {
                "model": key[0], "domain": key[1], "variant_type": key[2],
                "cosine_ci95_low": cosine_low, "cosine_ci95_high": cosine_high,
                "recall_at_1_ci95_low": r1_low, "recall_at_1_ci95_high": r1_high,
            }
        )
    return summary.merge(pd.DataFrame(ci_rows), on=group_columns, how="left")


def contrast_scores(details: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Compute preserving-vs-changed sensitivity margins for matched samples."""
    rows: list[dict] = []
    indexed = details.set_index(["model", "sample_id", "variant_type"], drop=False)
    models = details["model"].unique()
    sample_ids = details["sample_id"].unique()
    for spec in specs:
        reference = str(spec["reference"])
        changed = str(spec["changed"])
        label = str(spec.get("label", f"{reference}_minus_{changed}"))
        for model in models:
            for sample_id in sample_ids:
                ref_key = (model, sample_id, reference)
                changed_key = (model, sample_id, changed)
                if ref_key not in indexed.index or changed_key not in indexed.index:
                    continue
                ref = indexed.loc[ref_key]
                alt = indexed.loc[changed_key]
                if isinstance(ref, pd.DataFrame) or isinstance(alt, pd.DataFrame):
                    raise ValueError("Contrast inputs must be unique per model/sample/variant")
                rows.append(
                    {
                        "model": model, "sample_id": sample_id, "domain": ref["domain"],
                        "contrast": label, "reference_variant": reference, "changed_variant": changed,
                        "reference_cosine": ref["cosine"], "changed_cosine": alt["cosine"],
                        "delta_cosine": ref["cosine"] - alt["cosine"],
                    }
                )
    return pd.DataFrame(rows)

