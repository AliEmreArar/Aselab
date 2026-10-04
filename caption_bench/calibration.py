"""Caption discrimination relative to same-domain competitors and paired controls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_jsonl


def negative_metrics(target, negatives):
    negatives = np.asarray(negatives, dtype=float)
    # Ties receive half credit. This is a pairwise win rate, not a calibrated probability.
    wins = (target > negatives).astype(float) + 0.5 * (target == negatives)
    return {
        "negative_percentile": float(wins.mean()),
        "negative_median": float(np.median(negatives)),
        "negative_p95": float(np.quantile(negatives, .95)),
        "median_margin": float(target - np.median(negatives)),
        "hardest_margin": float(target - negatives.max()),
    }


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    means = np.random.default_rng(42).choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [.025, .975])


def analyze(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    variants = read_jsonl(run / "variants.jsonl")
    rows, control_rows = [], []
    for model in manifest["models"]:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
        with np.load(run / "embeddings" / f"{safe}.npz", allow_pickle=False) as z:
            ids = [str(x) for x in z["caption_ids"]]
            vids = [str(x) for x in z["variant_ids"]]
            originals = z["originals"].astype(float)
            vectors = z["variants"].astype(float)
        originals /= np.maximum(np.linalg.norm(originals, axis=1, keepdims=True), 1e-12)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        vi = {sid: i for i, sid in enumerate(vids)}
        for variant in variants:
            sid, typ = variant["sample_id"], variant["variant_type"]
            if typ == "identity":
                continue
            target_index = ids.index(sid)
            candidates = [i for i, key in enumerate(ids) if key.split("::")[0] == variant["domain"]]
            others = [i for i in candidates if i != target_index]
            query = vectors[vi[f"{sid}::{typ}"]]
            scores = originals @ query
            correct = float(scores[target_index])
            # Fixed per-anchor, per-model hard set: nearest 5 original captions.
            anchor_scores = originals[others] @ originals[target_index]
            hard = np.asarray(others)[np.argsort(-anchor_scores, kind="stable")[:5]]
            hardest = others[int(np.argmax(scores[others]))]
            row = {"model": model, "domain": variant["domain"], "sample_id": sid,
                   "variant_type": typ, "relation": variant["relation"], "cosine": correct,
                   # Conservative tie handling: rank after every tied competitor.
                   "rank": 1 + int(np.sum(scores[others] >= correct)),
                   "negative_count": len(others), "hard_negative_count": len(hard),
                   "hard_set_margin": correct - float(scores[hard].max()),
                   "hard_negative_win_rate": negative_metrics(correct, scores[hard])["negative_percentile"],
                   "hardest_sample_id": ids[hardest], "hardest_cosine": float(scores[hardest]),
                   **negative_metrics(correct, scores[others])}
            row["recall_at_1"] = float(row["rank"] == 1 and row["hardest_margin"] > 0)
            row["reciprocal_rank"] = 1. / row["rank"]
            rows.append(row)
            if typ == "control_one_fact":
                base = vectors[vi[f"{sid}::llm_balanced"]]
                preserving_key = f"{sid}::control_synonym"
                if preserving_key in vi:
                    preserving = vectors[vi[preserving_key]]
                    preserved_score, changed_score = float(base @ preserving), float(base @ query)
                    gap = preserved_score - changed_score
                    control_rows.append({"model": model, "domain": variant["domain"], "sample_id": sid,
                                         "preserving_cosine": preserved_score, "changed_cosine": changed_score,
                                         "preserving_minus_changed": gap,
                                         "preserving_wins": float(gap > 1e-7),
                                         "tie": float(abs(gap) <= 1e-7)})
    details = pd.DataFrame(rows)
    details.to_csv(run / "discrimination_details.csv", index=False)
    summaries = []
    for (model, domain, typ), group in details.groupby(["model", "domain", "variant_type"]):
        low, high = bootstrap(group["hardest_margin"])
        summaries.append({"model": model, "domain": domain, "variant_type": typ, "n": len(group),
                          **{key: float(group[key].mean()) for key in
                             ("cosine", "negative_percentile", "median_margin", "hardest_margin",
                              "hard_set_margin", "hard_negative_win_rate", "recall_at_1", "reciprocal_rank")},
                          "margin_ci_low": float(low), "margin_ci_high": float(high)})
    pd.DataFrame(summaries).to_csv(run / "discrimination_summary.csv", index=False)
    controls = pd.DataFrame(control_rows)
    controls.to_csv(run / "control_sensitivity_details.csv", index=False)
    if not controls.empty:
        controls.groupby(["model", "domain"]).agg(
            n=("sample_id", "size"), mean_gap=("preserving_minus_changed", "mean"),
            preserving_win_rate=("preserving_wins", "mean"), tie_rate=("tie", "mean")
        ).reset_index().to_csv(run / "control_sensitivity_summary.csv", index=False)
    return details
