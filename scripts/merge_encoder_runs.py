"""Merge independently executed encoder runs that used identical captions/variants."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from caption_bench.report import build_report


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _agreement(output: Path, model_names: list[str], k: int) -> pd.DataFrame:
    embeddings: dict[str, np.ndarray] = {}
    caption_ids: np.ndarray | None = None
    for name in model_names:
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
        with np.load(output / "embeddings" / f"{safe}.npz", allow_pickle=False) as cache:
            current_ids = cache["caption_ids"]
            if caption_ids is None:
                caption_ids = current_ids
            elif not np.array_equal(caption_ids, current_ids):
                raise ValueError(f"Caption order differs for {name}")
            vectors = cache["originals"].astype(np.float64)
            embeddings[name] = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    rows: list[dict] = []
    count = len(caption_ids) if caption_ids is not None else 0
    k = min(k, max(1, count - 1))
    for left_index, left_name in enumerate(model_names):
        left_sim = embeddings[left_name] @ embeddings[left_name].T
        for right_name in model_names[left_index + 1:]:
            right_sim = embeddings[right_name] @ embeddings[right_name].T
            upper = np.triu_indices(count, k=1)
            left_ranks = pd.Series(left_sim[upper]).rank(method="average").to_numpy()
            right_ranks = pd.Series(right_sim[upper]).rank(method="average").to_numpy()
            overlaps = []
            for row_index in range(count):
                left_order = [int(i) for i in np.argsort(-left_sim[row_index]) if i != row_index][:k]
                right_order = [int(i) for i in np.argsort(-right_sim[row_index]) if i != row_index][:k]
                overlaps.append(len(set(left_order) & set(right_order)) / k)
            rows.append({
                "model_a": left_name, "model_b": right_name,
                "spearman": float(np.corrcoef(left_ranks, right_ranks)[0, 1]),
                "knn_k": k, "knn_overlap": float(np.mean(overlaps)), "pairs": len(upper[0]),
            })
    return pd.DataFrame(rows)


def merge(base: Path, additions: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    base_variants = _read_jsonl(base / "variants.jsonl")
    shutil.copytree(base, output)
    manifests = [json.loads((base / "manifest.json").read_text(encoding="utf-8"))]
    details = [pd.read_csv(base / "details.csv")]
    summaries = [pd.read_csv(base / "summary.csv")]
    diagnostics = [pd.read_csv(base / "embedding_diagnostics.csv")]
    for addition in additions:
        if _read_jsonl(addition / "variants.jsonl") != base_variants:
            raise ValueError(f"Variant rows differ: {addition}")
        manifest = json.loads((addition / "manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        details.append(pd.read_csv(addition / "details.csv"))
        summaries.append(pd.read_csv(addition / "summary.csv"))
        diagnostics.append(pd.read_csv(addition / "embedding_diagnostics.csv"))
        for model in manifest["models"]:
            safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in model)
            shutil.copy2(addition / "embeddings" / f"{safe}.npz", output / "embeddings" / f"{safe}.npz")
    pd.concat(details, ignore_index=True).to_csv(output / "details.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(output / "embedding_diagnostics.csv", index=False)
    manifest = manifests[0]
    manifest["name"] = "multimodal-and-text-encoder-expansion-pilot-v3"
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["models"] = [model for item in manifests for model in item["models"]]
    manifest["config"]["models"] = [model for item in manifests for model in item["config"]["models"]]
    manifest["notes"].append(
        "Mamba-3 SISO is a raw causal-LM last-token baseline, not a sentence-embedding or vision-aligned model."
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _agreement(output, manifest["models"], int(manifest["config"].get("agreement_k", 10))).to_csv(
        output / "encoder_agreement.csv", index=False
    )
    build_report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--addition", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge(args.base.resolve(), [path.resolve() for path in args.addition], args.output.resolve())
    print(args.output.resolve())
