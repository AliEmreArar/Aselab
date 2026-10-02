from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from .data import load_captions, load_variants, read_jsonl, write_jsonl
from .encoders import create_encoder
from .metrics import contrast_scores, score_variants, summarize
from .report import build_report
from .transforms import generate_variants


def run_experiment(config_path: str | Path) -> Path:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    base = config_file.parent

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()

    captions = load_captions(resolve(config["dataset"]))
    seed = int(config.get("seed", 42))
    variants = generate_variants(captions, list(config.get("transforms", [])), seed)
    variant_files = [resolve(path) for path in config.get("variant_files", [])]
    variants.extend(load_variants(variant_files))
    # LLM variants are tied to the exact source caption used during generation.
    originals = {c.sample_id: c.text for c in captions}
    for path in variant_files:
        for row in read_jsonl(path):
            if row.get("source_sha256") and row["sample_id"] in originals:
                source_hash = hashlib.sha256(originals[row["sample_id"]].encode("utf-8")).hexdigest()
                if source_hash != row["source_sha256"]:
                    raise ValueError(f"Source caption changed for {row['sample_id']}; regenerate its LLM variants.")
    if not variants:
        raise ValueError("No variants configured. Add transforms or variant_files.")

    known_ids = {caption.sample_id for caption in captions}
    unknown = sorted({variant.sample_id for variant in variants} - known_ids)
    if unknown:
        raise ValueError(f"Variants refer to unknown sample IDs: {unknown[:5]}")
    present = {(variant.sample_id, variant.variant_type) for variant in variants}
    if len(present) != len(variants):
        raise ValueError("Each (sample_id, variant_type) pair must be unique")

    output_dir = resolve(str(config.get("output_dir", "../runs/latest")))
    output_dir.mkdir(parents=True, exist_ok=True)
    caption_by_id = {caption.sample_id: caption for caption in captions}
    write_jsonl(
        output_dir / "variants.jsonl",
        (
            {
                "sample_id": variant.sample_id,
                "domain": caption_by_id[variant.sample_id].domain,
                "split": caption_by_id[variant.sample_id].split,
                "variant_type": variant.variant_type,
                "text": variant.text,
                "generator": variant.generator,
                "relation": variant.relation,
            }
            for variant in variants
        ),
    )
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    all_frames: list[pd.DataFrame] = []
    original_embeddings: dict[str, np.ndarray] = {}
    original_texts = [caption.text for caption in captions]
    variant_texts = [variant.text for variant in variants]
    for model_spec in config["models"]:
        encoder = create_encoder(dict(model_spec))
        original_batch = encoder.encode(original_texts)
        variant_batch = encoder.encode(variant_texts)
        original_embeddings[encoder.name] = original_batch.embeddings
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in encoder.name)
        np.savez_compressed(
            embedding_dir / f"{safe_name}.npz",
            originals=original_batch.embeddings,
            variants=variant_batch.embeddings,
            caption_ids=np.asarray([caption.sample_id for caption in captions]),
            variant_ids=np.asarray([f"{variant.sample_id}::{variant.variant_type}" for variant in variants]),
        )
        all_frames.append(
            score_variants(encoder.name, captions, variants, original_batch, variant_batch)
        )

    details = pd.concat(all_frames, ignore_index=True)
    summary = summarize(details)
    details.to_csv(output_dir / "details.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    contrast = contrast_scores(details, list(config.get("contrasts", [])))
    if not contrast.empty:
        contrast.to_csv(output_dir / "contrasts.csv", index=False)
        contrast_summary = (
            contrast.groupby(["model", "domain", "contrast"])
            .agg(n=("sample_id", "size"), delta_cosine_mean=("delta_cosine", "mean"),
                 delta_cosine_std=("delta_cosine", "std"), positive_rate=("delta_cosine", lambda x: (x > 0).mean()))
            .reset_index()
        )
        contrast_summary.to_csv(output_dir / "contrast_summary.csv", index=False)

    agreement_rows: list[dict] = []
    model_names = list(original_embeddings)
    k = min(int(config.get("agreement_k", 10)), max(1, len(captions) - 1))
    for left_idx, left_name in enumerate(model_names):
        left_sim = original_embeddings[left_name] @ original_embeddings[left_name].T
        for right_name in model_names[left_idx + 1 :]:
            right_sim = original_embeddings[right_name] @ original_embeddings[right_name].T
            upper = np.triu_indices(len(captions), k=1)
            left_ranks = pd.Series(left_sim[upper]).rank(method="average").to_numpy()
            right_ranks = pd.Series(right_sim[upper]).rank(method="average").to_numpy()
            spearman = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
            overlaps: list[float] = []
            for row_idx in range(len(captions)):
                left_order = np.argsort(-left_sim[row_idx])
                right_order = np.argsort(-right_sim[row_idx])
                left_top = set([int(idx) for idx in left_order if idx != row_idx][:k])
                right_top = set([int(idx) for idx in right_order if idx != row_idx][:k])
                overlaps.append(len(left_top & right_top) / k)
            agreement_rows.append(
                {"model_a": left_name, "model_b": right_name, "spearman": spearman,
                 "knn_k": k, "knn_overlap": float(np.mean(overlaps)), "pairs": len(upper[0])}
            )
    pd.DataFrame(agreement_rows, columns=["model_a", "model_b", "spearman", "knn_k", "knn_overlap", "pairs"]).to_csv(
        output_dir / "encoder_agreement.csv", index=False
    )

    diagnostic_rows: list[dict] = []
    labels = np.asarray([caption.domain for caption in captions])
    for model_name, embeddings in original_embeddings.items():
        similarity = embeddings @ embeddings.T
        distances = 1.0 - similarity
        silhouettes: list[float] = []
        for idx, label in enumerate(labels):
            same = np.where(labels == label)[0]
            same = same[same != idx]
            a = float(distances[idx, same].mean()) if len(same) else 0.0
            other_means = [float(distances[idx, labels == other].mean()) for other in set(labels) if other != label]
            b = min(other_means)
            silhouettes.append((b - a) / max(a, b, 1e-12))
        for domain in sorted(set(labels)):
            indices = np.where(labels == domain)[0]
            domain_sim = similarity[np.ix_(indices, indices)]
            upper = domain_sim[np.triu_indices(len(indices), k=1)]
            diagnostic_rows.append(
                {"model": model_name, "domain": domain, "within_domain_cosine_mean": float(upper.mean()),
                 "within_domain_cosine_std": float(upper.std()),
                 "global_domain_silhouette": float(np.mean(silhouettes))}
            )
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "embedding_diagnostics.csv", index=False)
    manifest = {
        "name": str(config.get("name", output_dir.name)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "caption_count": len(captions),
        "variant_count": len(variants),
        "models": [model["name"] for model in config["models"]],
        "notes": [
            "Retrieval ranking is computed against original captions within the same domain.",
            "Cosine scales are not calibrated across model families; compare ranks, margins and relative changes.",
            "word_dropout is a stress test and is not guaranteed to preserve all identity evidence.",
            "Bootstrap intervals use 2,000 deterministic resamples per model/domain/variant group.",
            "encoder_agreement.csv compares the complete 150x150 original-caption similarity geometry.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(output_dir)
    return output_dir

