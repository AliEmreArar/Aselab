from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Caption:
    sample_id: str
    domain: str
    split: str
    filename: str
    text: str
    image: str | None = None


@dataclass(frozen=True)
class Variant:
    sample_id: str
    variant_type: str
    text: str
    generator: str = "external"
    relation: str = "unspecified"


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_captions(path: str | Path) -> list[Caption]:
    captions: list[Caption] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        missing = {"domain", "split", "filename", "caption"} - row.keys()
        if missing:
            raise ValueError(f"Caption row is missing fields {sorted(missing)}: {row}")
        sample_id = str(row.get("sample_id") or f"{row['domain']}::{row['split']}::{row['filename']}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        captions.append(
            Caption(
                sample_id=sample_id,
                domain=str(row["domain"]),
                split=str(row["split"]),
                filename=str(row["filename"]),
                text=str(row["caption"]).strip(),
                image=row.get("image"),
            )
        )
    if not captions:
        raise ValueError(f"No captions found in {path}")
    return captions


def load_variants(paths: Iterable[str | Path]) -> list[Variant]:
    variants: list[Variant] = []
    keys: set[tuple[str, str]] = set()
    for path in paths:
        for row in read_jsonl(path):
            missing = {"sample_id", "variant_type", "text"} - row.keys()
            if missing:
                raise ValueError(f"Variant row is missing fields {sorted(missing)}: {row}")
            key = (str(row["sample_id"]), str(row["variant_type"]))
            if key in keys:
                raise ValueError(f"Duplicate variant {key} across variant files")
            keys.add(key)
            variants.append(
                Variant(
                    sample_id=key[0],
                    variant_type=key[1],
                    text=str(row["text"]).strip(),
                    generator=str(row.get("generator", "external")),
                    relation=str(row.get("relation", "unspecified")),
                )
            )
    return variants

