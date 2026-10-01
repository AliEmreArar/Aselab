from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EncodingBatch:
    embeddings: np.ndarray
    token_counts: list[int]
    truncated: list[bool]


def _normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


class TextEncoder(ABC):
    def __init__(self, spec: dict):
        self.spec = spec
        self.name = str(spec["name"])
        self.prefix = str(spec.get("text_prefix", ""))

    @abstractmethod
    def encode(self, texts: list[str]) -> EncodingBatch:
        raise NotImplementedError


class HashingEncoder(TextEncoder):
    """Offline lexical baseline. It is intentionally not a semantic model."""

    def encode(self, texts: list[str]) -> EncodingBatch:
        dim = int(self.spec.get("dimension", 768))
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        counts: list[int] = []
        for row, raw_text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", (self.prefix + raw_text).lower())
            counts.append(len(tokens))
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                vectors[row, value % dim] += 1.0 if (value >> 63) == 0 else -1.0
        return EncodingBatch(_normalize(vectors), counts, [False] * len(texts))


class TransformersEncoder(TextEncoder):
    def __init__(self, spec: dict, feature_mode: bool):
        super().__init__(spec)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.feature_mode = feature_mode
        self.device = str(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(spec.get("batch_size", 32))
        self.model_id = str(spec["model_id"])
        hub_kwargs = {}
        if spec.get("revision"):
            hub_kwargs["revision"] = str(spec["revision"])
        if spec.get("trust_remote_code"):
            hub_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **hub_kwargs)
        model_kwargs = dict(hub_kwargs)
        if spec.get("use_safetensors") is not None:
            model_kwargs["use_safetensors"] = bool(spec["use_safetensors"])
        self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs).to(self.device).eval()
        configured_max = spec.get("max_length")
        tokenizer_max = int(getattr(self.tokenizer, "model_max_length", 512))
        if tokenizer_max > 1_000_000:
            tokenizer_max = 512
        self.max_length = int(configured_max or tokenizer_max)
        self.padding = str(spec.get("padding", "longest"))

    def _to_numpy(self, value) -> np.ndarray:
        if isinstance(value, self.torch.Tensor):
            return value.detach().float().cpu().numpy()
        for attr in ("text_embeds", "pooler_output"):
            candidate = getattr(value, attr, None)
            if candidate is not None:
                return candidate.detach().float().cpu().numpy()
        if isinstance(value, (tuple, list)) and value:
            return self._to_numpy(value[0])
        raise TypeError(f"Cannot extract embeddings from {type(value)!r}")

    def encode(self, texts: list[str]) -> EncodingBatch:
        all_embeddings: list[np.ndarray] = []
        all_counts: list[int] = []
        all_truncated: list[bool] = []
        prefixed = [self.prefix + text for text in texts]
        for start in range(0, len(prefixed), self.batch_size):
            batch = prefixed[start : start + self.batch_size]
            raw = self.tokenizer(batch, padding=False, truncation=False)
            raw_lengths = [len(ids) for ids in raw["input_ids"]]
            inputs = self.tokenizer(
                batch,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                if self.feature_mode:
                    features = self.model.get_text_features(**inputs)
                    vectors = self._to_numpy(features)
                else:
                    output = self.model(**inputs)
                    hidden = output.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    vectors = pooled.detach().float().cpu().numpy()
            all_embeddings.append(vectors)
            all_counts.extend(raw_lengths)
            all_truncated.extend(length > self.max_length for length in raw_lengths)
        return EncodingBatch(_normalize(np.vstack(all_embeddings)), all_counts, all_truncated)


class SentenceTransformersEncoder(TextEncoder):
    def __init__(self, spec: dict):
        super().__init__(spec)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The sentence_transformers backend needs the optional dependency: "
                "pip install -e '.[sentence-transformers]'"
            ) from exc
        import torch

        self.batch_size = int(spec.get("batch_size", 32))
        self.max_length = spec.get("max_length")
        device = str(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SentenceTransformer(str(spec["model_id"]), device=device)
        if self.max_length is not None:
            self.model.max_seq_length = int(self.max_length)

    def encode(self, texts: list[str]) -> EncodingBatch:
        prefixed = [self.prefix + text for text in texts]
        tokenizer = self.model.tokenizer
        raw = tokenizer(prefixed, padding=False, truncation=False)
        counts = [len(ids) for ids in raw["input_ids"]]
        max_length = int(self.model.max_seq_length)
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return EncodingBatch(np.asarray(embeddings, dtype=np.float32), counts, [n > max_length for n in counts])


def create_encoder(spec: dict) -> TextEncoder:
    backend = str(spec.get("backend", "hash"))
    if backend == "hash":
        return HashingEncoder(spec)
    if backend == "hf_mean_pool":
        return TransformersEncoder(spec, feature_mode=False)
    if backend == "hf_text_features":
        return TransformersEncoder(spec, feature_mode=True)
    if backend == "sentence_transformers":
        return SentenceTransformersEncoder(spec)
    raise ValueError(
        f"Unknown backend '{backend}'. Expected hash, hf_mean_pool, hf_text_features, or sentence_transformers."
    )

