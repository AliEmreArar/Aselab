"""Measure non-identical original-caption similarities within one domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", default="caption_samples_50/all_samples.jsonl", type=Path)
    parser.add_argument("--domain", default="face")
    parser.add_argument("--fixed-a")
    parser.add_argument("--fixed-b")
    args = parser.parse_args()

    run = args.run.resolve()
    samples = [json.loads(line) for line in args.dataset.resolve().read_text(encoding="utf-8").splitlines()]
    sample_ids = [
        f"{row['domain']}::{row['split']}::{row['filename']}"
        for row in samples if row["domain"] == args.domain
    ]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model in manifest["models"]:
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in model)
        with np.load(run / "embeddings" / f"{safe}.npz", allow_pickle=False) as cache:
            indices = {str(sample_id): idx for idx, sample_id in enumerate(cache["caption_ids"])}
            vectors = cache["originals"][[indices[sample_id] for sample_id in sample_ids]].astype(np.float64)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarity = vectors @ vectors.T
        upper = np.triu_indices(len(sample_ids), k=1)
        values = similarity[upper]
        order = np.argsort(-values)
        for rank, pair_index in enumerate(order, start=1):
            left, right = upper[0][pair_index], upper[1][pair_index]
            pair_rows.append({
                "model": model, "domain": args.domain, "pair_rank": rank,
                "sample_a": sample_ids[left], "sample_b": sample_ids[right],
                "cosine": float(values[pair_index]),
            })
        top_index = order[0]
        top_left, top_right = upper[0][top_index], upper[1][top_index]
        fixed_score = float("nan")
        if args.fixed_a and args.fixed_b:
            fixed_score = float(similarity[sample_ids.index(args.fixed_a), sample_ids.index(args.fixed_b)])
        summary_rows.append({
            "model": model, "domain": args.domain, "pair_count": len(values),
            "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values[top_index]),
            "share_ge_0_90": float(np.mean(values >= 0.90)),
            "share_ge_0_95": float(np.mean(values >= 0.95)),
            "top_sample_a": sample_ids[top_left], "top_sample_b": sample_ids[top_right],
            "fixed_sample_a": args.fixed_a or "", "fixed_sample_b": args.fixed_b or "",
            "fixed_pair_cosine": fixed_score,
        })
    pd.DataFrame(pair_rows).to_csv(run / f"different_{args.domain}_original_pairs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(run / f"different_{args.domain}_original_pair_summary.csv", index=False)
    print(run / f"different_{args.domain}_original_pair_summary.csv")


if __name__ == "__main__":
    main()
