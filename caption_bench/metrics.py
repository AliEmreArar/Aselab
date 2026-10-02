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
                "original_overflowed": original_batch.overflowed[target_idx],
                "variant_overflowed": variant_batch.overflowed[variant_idx],
                "original_chunks": original_batch.chunk_counts[target_idx],
                "variant_chunks": variant_batch.chunk_counts[variant_idx],
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


def score_image_retrieval(
    model_name: str,
    captions: list[Caption],
    variants: list[Variant],
    variant_batch: EncodingBatch,
    image_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Rank the matching image for each caption variant within its domain."""
    caption_index = {caption.sample_id: idx for idx, caption in enumerate(captions)}
    domains: dict[str, list[int]] = {}
    for idx, caption in enumerate(captions):
        domains.setdefault(caption.domain, []).append(idx)
    rows: list[dict] = []
    for variant_idx, variant in enumerate(variants):
        target_idx = caption_index[variant.sample_id]
        caption = captions[target_idx]
        candidate_indices = domains[caption.domain]
        scores = variant_batch.embeddings[variant_idx] @ image_embeddings[candidate_indices].T
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
                "caption_text": variant.text,
                "image_cosine": target_score,
                "image_rank": rank,
                "image_reciprocal_rank": 1.0 / rank,
                "image_recall_at_1": int(rank <= 1),
                "image_recall_at_5": int(rank <= 5),
                "image_hardest_negative_cosine": hardest_negative,
                "image_margin": target_score - hardest_negative,
                "image_pool_size": len(candidate_indices),
            }
        )
    return pd.DataFrame(rows)


def summarize_image_retrieval(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    identity = details[details["variant_type"] == "identity"].set_index(["model", "sample_id"])
    rows: list[dict] = []
    for (model, domain, variant_type), group in details.groupby(
        ["model", "domain", "variant_type"], dropna=False, sort=True
    ):
        if variant_type == "identity":
            baseline = group
        else:
            keys = [(model, sample_id) for sample_id in group["sample_id"]]
            baseline = identity.loc[keys].reset_index()
        row = {
            "model": model,
            "domain": domain,
            "variant_type": variant_type,
            "n": len(group),
            "image_cosine_mean": group["image_cosine"].mean(),
            "image_margin_mean": group["image_margin"].mean(),
            "image_mrr": group["image_reciprocal_rank"].mean(),
            "image_recall_at_1": group["image_recall_at_1"].mean(),
            "image_recall_at_5": group["image_recall_at_5"].mean(),
            "matched_identity_cosine_mean": baseline["image_cosine"].mean(),
            "matched_identity_margin_mean": baseline["image_margin"].mean(),
            "matched_identity_mrr": baseline["image_reciprocal_rank"].mean(),
            "matched_identity_recall_at_1": baseline["image_recall_at_1"].mean(),
            "matched_identity_recall_at_5": baseline["image_recall_at_5"].mean(),
        }
        row["delta_image_cosine"] = row["image_cosine_mean"] - row["matched_identity_cosine_mean"]
        row["delta_image_margin"] = row["image_margin_mean"] - row["matched_identity_margin_mean"]
        row["delta_image_mrr"] = row["image_mrr"] - row["matched_identity_mrr"]
        row["delta_image_recall_at_1"] = row["image_recall_at_1"] - row["matched_identity_recall_at_1"]
        row["delta_image_recall_at_5"] = row["image_recall_at_5"] - row["matched_identity_recall_at_5"]
        rows.append(row)
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
    details = details.copy()
    for target, fallback in (
        ("original_overflowed", "original_truncated"),
        ("variant_overflowed", "variant_truncated"),
    ):
        if target not in details:
            details[target] = details[fallback]
    for column in ("original_chunks", "variant_chunks"):
        if column not in details:
            details[column] = 1
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
            original_overflow_rate=("original_overflowed", "mean"),
            variant_overflow_rate=("variant_overflowed", "mean"),
            original_chunks_mean=("original_chunks", "mean"),
            variant_chunks_mean=("variant_chunks", "mean"),
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

