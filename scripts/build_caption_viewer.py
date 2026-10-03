"""Build an offline image/caption explorer from the recorded benchmark runs."""
import base64
import argparse
import csv
import json
import mimetypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(*, generated_path=None, gemini_folder=None, output_path=None):
    samples = []
    for line in (ROOT / "caption_samples_50/all_samples.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        path = ROOT / "caption_samples_50" / row["image"]
        assert path.is_file(), path
        samples.append({
            "id": f'{row["domain"]}::{row["split"]}::{row["filename"]}',
            "domain": row["domain"], "split": row["split"], "filename": row["filename"],
            "original": row["caption"],
            "image": f'data:{mimetypes.guess_type(path.name)[0]};base64,' + base64.b64encode(path.read_bytes()).decode(),
        })
    runs = {}
    generated_paths = [Path(generated_path)] if generated_path else [ROOT / "variants/gemini/variants.jsonl", ROOT / "variants/gemini_human/variants.jsonl"]
    generated = [json.loads(line) for path in generated_paths if path.exists() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated_lookup = {(row["sample_id"], row["variant_type"]): row for row in generated}
    run_names = ["priority", "clip_b16", "quick"]
    gemini_folder = Path(gemini_folder) if gemini_folder else ROOT / "runs/gemini"
    if (gemini_folder / "manifest.json").exists():
        run_names.append("gemini")
    for name in run_names:
        folder = gemini_folder if name == "gemini" else ROOT / "runs" / name
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        variants = {}
        for line in (folder / "variants.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            source = generated_lookup.get((row["sample_id"], row["variant_type"]), {})
            metadata = {key: source[key] for key in ("change_note", "retained_facts", "omitted_facts", "generator", "review_warnings", "word_count", "target_words") if key in source} if source.get("text") == row["text"] else {}
            variants.setdefault(row["sample_id"], []).append({
                "type": row["variant_type"], "text": row["text"], "relation": row["relation"], "scores": {},
                **metadata,
            })
        lookup = {(sid, v["type"]): v for sid, group in variants.items() for v in group}
        with (folder / "details.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                score = {key: float(row[key]) for key in ("cosine", "margin", "hardest_negative_cosine")}
                score.update({key: int(row[key]) for key in ("rank", "original_tokens", "variant_tokens", "retrieval_pool_size")})
                score.update({key: row[key].lower() == "true" for key in ("original_truncated", "variant_truncated")})
                score.update({key: int(row[key]) for key in ("original_chunks", "variant_chunks") if key in row})
                score.update({key: row[key].lower() == "true" for key in ("original_overflowed", "variant_overflowed") if key in row})
                lookup[row["sample_id"], row["variant_type"]]["scores"][row["model"]] = score
        image_details_path = folder / "image_retrieval_details.csv"
        if image_details_path.exists():
            with image_details_path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    score = lookup[row["sample_id"], row["variant_type"]]["scores"][row["model"]]
                    score.update({key: float(row[key]) for key in ("image_cosine", "image_margin")})
                    score.update({key: int(row[key]) for key in ("image_rank", "image_pool_size")})
        pairs = {sample["id"]: {} for sample in samples}
        for model in manifest["models"]:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
            with np.load(folder / "embeddings" / f"{safe_name}.npz", allow_pickle=False) as cache:
                indices = {str(key): i for i, key in enumerate(cache["variant_ids"])}
                for sid, group in variants.items():
                    vectors = cache["variants"][[indices[f'{sid}::{v["type"]}'] for v in group]].astype(np.float64)
                    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                    matrix = vectors @ vectors.T
                    assert np.isfinite(matrix).all()
                    pairs[sid][model] = matrix.round(7).tolist()
                    # Confirm cached vectors agree with the recorded original-vs-variant scores.
                    identity = next(i for i, v in enumerate(group) if v["type"] == "identity")
                    for i, v in enumerate(group):
                        assert abs(matrix[identity, i] - v["scores"][model]["cosine"]) < 2e-5
        pair_summary = []
        pair_summary_path = folder / "different_face_original_pair_summary.csv"
        if pair_summary_path.exists():
            with pair_summary_path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    pair_summary.append({
                        **{key: row[key] for key in ("model", "top_sample_a", "top_sample_b", "fixed_sample_a", "fixed_sample_b")},
                        **{key: float(row[key]) for key in ("median", "p95", "maximum", "share_ge_0_90", "share_ge_0_95", "fixed_pair_cosine")},
                        "pair_count": int(row["pair_count"]),
                    })
        runs[name] = {
            "name": manifest.get("name", name),
            "models": manifest["models"],
            "date": manifest["created_at"],
            "variants": variants,
            "pairs": pairs,
            "differentFaceStats": pair_summary,
        }
    if generated:
        scored = runs.get("gemini", {}).get("variants", {})
        all_scored = all(any(v["type"] == row["variant_type"] and v["text"] == row["text"] for v in scored.get(row["sample_id"], [])) for row in generated)
        if not all_scored:
            preview = {s["id"]: [{"type": "identity", "text": s["original"], "relation": "same", "scores": {}}] for s in samples}
            for row in generated:
                if row["sample_id"] not in preview:
                    raise ValueError(f"Generated caption references unknown image: {row['sample_id']}")
                preview[row["sample_id"]].append({"type": row["variant_type"], "text": row["text"], "relation": row["relation"],
                    "scores": {}, **{key: row[key] for key in ("change_note", "retained_facts", "omitted_facts", "generator", "review_warnings", "word_count", "target_words") if key in row}})
            runs["gemini_pending"] = {"models": [], "variants": preview, "pairs": {}, "date": ""}
    payload = json.dumps({"samples": samples, "runs": runs}, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = Path(__file__).with_name("caption_viewer.html").read_text(encoding="utf-8")
    output = Path(output_path) if output_path else ROOT / "caption_inceleme.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"Created {output} ({output.stat().st_size:,} bytes), {len(samples)} images; validated all pair scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline image/caption viewer")
    parser.add_argument("--generated-variants", help="Gemini JSONL file (defaults to variants/gemini/variants.jsonl)")
    parser.add_argument("--gemini-run-dir", help="Scored Gemini run folder (defaults to runs/gemini)")
    parser.add_argument("--output", help="Output HTML path")
    args = parser.parse_args()
    build(generated_path=args.generated_variants, gemini_folder=args.gemini_run_dir, output_path=args.output)
