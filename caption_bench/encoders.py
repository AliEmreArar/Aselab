from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EncodingBatch:
    embeddings: np.ndarray
    token_counts: list[int]
    truncated: list[bool]
    chunk_counts: list[int] | None = None
    overflowed: list[bool] | None = None

    def __post_init__(self) -> None:
        size = len(self.token_counts)
        if self.chunk_counts is None:
            self.chunk_counts = [1] * size
        if self.overflowed is None:
            self.overflowed = list(self.truncated)


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

    @property
    def supports_images(self) -> bool:
        return False

    def encode_images(self, paths: list[str | Path]) -> np.ndarray:
        raise TypeError(f"{self.name} does not expose an image encoder")


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
        return EncodingBatch(
            _normalize(vectors), counts, [False] * len(texts),
            chunk_counts=[1] * len(texts), overflowed=[False] * len(texts),
        )


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
        hub_kwargs: dict = {}
        if spec.get("revision"):
            hub_kwargs["revision"] = str(spec["revision"])
        if spec.get("trust_remote_code"):
            hub_kwargs["trust_remote_code"] = True
        tokenizer_kwargs = dict(hub_kwargs)
        if spec.get("tokenizer_use_fast") is not None:
            tokenizer_kwargs["use_fast"] = bool(spec["tokenizer_use_fast"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        self.hub_kwargs = hub_kwargs
        model_kwargs = dict(hub_kwargs)
        if spec.get("code_revision"):
            model_kwargs["code_revision"] = str(spec["code_revision"])
        if spec.get("use_safetensors") is not None:
            model_kwargs["use_safetensors"] = bool(spec["use_safetensors"])
        dtype_name = spec.get("dtype")
        if dtype_name:
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            if dtype_name not in dtype_map:
                raise ValueError("dtype must be float16, bfloat16, or float32")
            model_kwargs["dtype"] = dtype_map[dtype_name]
        self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs).to(self.device).eval()
        self.compute_dtype = next(self.model.parameters()).dtype
        configured_max = spec.get("max_length")
        tokenizer_max = int(getattr(self.tokenizer, "model_max_length", 512))
        if tokenizer_max > 1_000_000:
            tokenizer_max = 512
        self.max_length = int(configured_max or tokenizer_max)
        self.padding = str(spec.get("padding", "longest"))
        self.long_text_strategy = str(spec.get("long_text_strategy", "truncate"))
        if self.long_text_strategy not in {"truncate", "chunk_weighted_mean"}:
            raise ValueError("long_text_strategy must be 'truncate' or 'chunk_weighted_mean'.")

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

    def _forward(self, inputs: dict) -> np.ndarray:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            if self.feature_mode:
                return self._to_numpy(self.model.get_text_features(**inputs))
            output = self.model(**inputs)
            hidden = output.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return pooled.detach().float().cpu().numpy()

    def _raw_token_ids(self, texts: list[str]) -> tuple[list[list[int]], int]:
        raw = self.tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        ids = raw["input_ids"]
        if ids and isinstance(ids[0], int):
            ids = [ids]
        special = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        return ids, special

    def _wrap_with_special_tokens(self, content_ids: list[int]) -> list[int]:
        """Apply the tokenizer's single-sequence template without decoding a chunk."""
        if not hasattr(self, "_special_prefix"):
            probe_raw = self.tokenizer("caption", add_special_tokens=False)["input_ids"]
            probe_full = self.tokenizer("caption", add_special_tokens=True)["input_ids"]
            for start in range(len(probe_full) - len(probe_raw) + 1):
                if probe_full[start : start + len(probe_raw)] == probe_raw:
                    self._special_prefix = probe_full[:start]
                    self._special_suffix = probe_full[start + len(probe_raw) :]
                    break
            else:
                raise RuntimeError(f"Cannot determine special-token template for {self.model_id}")
        return self._special_prefix + list(content_ids) + self._special_suffix

    def _encode_truncated(self, prefixed: list[str]) -> EncodingBatch:
        all_embeddings: list[np.ndarray] = []
        all_counts: list[int] = []
        all_truncated: list[bool] = []
        for start in range(0, len(prefixed), self.batch_size):
            batch = prefixed[start : start + self.batch_size]
            raw_ids, special = self._raw_token_ids(batch)
            raw_lengths = [len(ids) + special for ids in raw_ids]
            inputs = self.tokenizer(
                batch,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            all_embeddings.append(self._forward(inputs))
            all_counts.extend(raw_lengths)
            all_truncated.extend(length > self.max_length for length in raw_lengths)
        return EncodingBatch(
            _normalize(np.vstack(all_embeddings)), all_counts, all_truncated,
            chunk_counts=[1] * len(prefixed), overflowed=list(all_truncated),
        )

    def _encode_chunked(self, prefixed: list[str]) -> EncodingBatch:
        raw_ids, special = self._raw_token_ids(prefixed)
        capacity = self.max_length - special
        if capacity < 1:
            raise ValueError(f"max_length={self.max_length} leaves no room for content tokens")
        encoded_chunks: list[dict] = []
        owners: list[int] = []
        weights: list[int] = []
        chunk_counts: list[int] = []
        token_counts = [len(ids) + special for ids in raw_ids]
        overflowed = [count > self.max_length for count in token_counts]
        for owner, ids in enumerate(raw_ids):
            pieces = [ids[start : start + capacity] for start in range(0, len(ids), capacity)] or [[]]
            chunk_counts.append(len(pieces))
            for piece in pieces:
                input_ids = self._wrap_with_special_tokens(piece)
                encoded_chunks.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                owners.append(owner)
                weights.append(max(1, len(piece)))

        chunk_vectors: list[np.ndarray] = []
        for start in range(0, len(encoded_chunks), self.batch_size):
            rows = encoded_chunks[start : start + self.batch_size]
            target_length = self.max_length if self.padding == "max_length" else max(len(row["input_ids"]) for row in rows)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError(f"Tokenizer for {self.model_id} has no pad_token_id")
            input_rows: list[list[int]] = []
            mask_rows: list[list[int]] = []
            for row in rows:
                needed = target_length - len(row["input_ids"])
                if self.tokenizer.padding_side == "left":
                    input_rows.append([pad_id] * needed + row["input_ids"])
                    mask_rows.append([0] * needed + row["attention_mask"])
                else:
                    input_rows.append(row["input_ids"] + [pad_id] * needed)
                    mask_rows.append(row["attention_mask"] + [0] * needed)
            inputs = {
                "input_ids": self.torch.tensor(input_rows, dtype=self.torch.long),
            }
            if "attention_mask" in self.tokenizer.model_input_names or not self.feature_mode:
                inputs["attention_mask"] = self.torch.tensor(mask_rows, dtype=self.torch.long)
            chunk_vectors.append(_normalize(self._forward(inputs)))
        vectors = np.vstack(chunk_vectors)
        combined = np.zeros((len(prefixed), vectors.shape[1]), dtype=np.float32)
        total_weights = np.zeros(len(prefixed), dtype=np.float32)
        for vector, owner, weight in zip(vectors, owners, weights):
            combined[owner] += vector * weight
            total_weights[owner] += weight
        combined /= np.clip(total_weights[:, None], 1e-12, None)
        return EncodingBatch(
            _normalize(combined), token_counts, [False] * len(prefixed),
            chunk_counts=chunk_counts, overflowed=overflowed,
        )

    def encode(self, texts: list[str]) -> EncodingBatch:
        prefixed = [self.prefix + text for text in texts]
        if self.long_text_strategy == "chunk_weighted_mean":
            return self._encode_chunked(prefixed)
        return self._encode_truncated(prefixed)

    @property
    def supports_images(self) -> bool:
        return self.feature_mode and hasattr(self.model, "get_image_features")

    def encode_images(self, paths: list[str | Path]) -> np.ndarray:
        if not self.supports_images:
            return super().encode_images(paths)
        from PIL import Image
        from transformers import AutoProcessor

        if not hasattr(self, "processor"):
            self.processor = AutoProcessor.from_pretrained(self.model_id, **self.hub_kwargs)
        vectors: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(self.device, dtype=self.compute_dtype) if value.is_floating_point() else value.to(self.device)
                for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                vectors.append(self._to_numpy(self.model.get_image_features(**inputs)))
        return _normalize(np.vstack(vectors))


class JinaClipEncoder(TextEncoder):
    """Adapter for jina-clip-v2's remote-code encode_text/encode_image API."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = str(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(spec.get("batch_size", 4))
        self.model_id = str(spec["model_id"])
        self.max_length = int(spec.get("max_length", 8192))
        self.truncate_dim = spec.get("truncate_dim")
        hub_kwargs = {"trust_remote_code": True}
        if spec.get("revision"):
            hub_kwargs["revision"] = str(spec["revision"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **hub_kwargs)
        model_kwargs = dict(hub_kwargs)
        if spec.get("code_revision"):
            model_kwargs["code_revision"] = str(spec["code_revision"])
        dtype_name = spec.get("dtype")
        if dtype_name:
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            if dtype_name not in dtype_map:
                raise ValueError("dtype must be float16, bfloat16, or float32")
            model_kwargs["dtype"] = dtype_map[dtype_name]
        self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs).to(self.device).eval()

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def encode(self, texts: list[str]) -> EncodingBatch:
        prefixed = [self.prefix + text for text in texts]
        raw = self.tokenizer(prefixed, padding=False, truncation=False)
        counts = [len(ids) for ids in raw["input_ids"]]
        vectors: list[np.ndarray] = []
        for start in range(0, len(prefixed), self.batch_size):
            batch = prefixed[start : start + self.batch_size]
            kwargs = {"truncate_dim": self.truncate_dim} if self.truncate_dim else {}
            vectors.append(self._as_numpy(self.model.encode_text(batch, **kwargs)))
        overflowed = [count > self.max_length for count in counts]
        return EncodingBatch(
            _normalize(np.vstack(vectors)), counts, list(overflowed),
            chunk_counts=[1] * len(texts), overflowed=overflowed,
        )

    @property
    def supports_images(self) -> bool:
        return True

    def encode_images(self, paths: list[str | Path]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        string_paths = [str(Path(path)) for path in paths]
        for start in range(0, len(string_paths), self.batch_size):
            batch = string_paths[start : start + self.batch_size]
            kwargs = {"truncate_dim": self.truncate_dim} if self.truncate_dim else {}
            vectors.append(self._as_numpy(self.model.encode_image(batch, **kwargs)))
        return _normalize(np.vstack(vectors))


class OpenClipEncoder(TextEncoder):
    """Adapter for OpenCLIP checkpoints such as EVA02-CLIP-L/14."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "The open_clip backend needs the optional dependency: pip install -e '.[open-clip]'"
            ) from exc
        import torch

        self.torch = torch
        self.device = str(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(spec.get("batch_size", 8))
        self.model_id = str(spec["model_id"])
        self.pretrained = str(spec["pretrained"])
        self.precision = str(spec.get("precision", "fp16" if self.device.startswith("cuda") else "fp32"))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, pretrained=self.pretrained, device=self.device, precision=self.precision
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.max_length = int(spec.get("max_length", getattr(self.model, "context_length", 77)))
        if str(spec.get("long_text_strategy", "truncate")) != "truncate":
            raise ValueError("open_clip currently supports only long_text_strategy=truncate")

    def _content_count(self, text: str) -> int:
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(text)
            return len(encoded.ids if hasattr(encoded, "ids") else encoded) + 2
        tokens = self.tokenizer([text], context_length=self.max_length)
        return int((tokens != 0).sum().item())

    def encode(self, texts: list[str]) -> EncodingBatch:
        prefixed = [self.prefix + text for text in texts]
        counts = [self._content_count(text) for text in prefixed]
        vectors: list[np.ndarray] = []
        for start in range(0, len(prefixed), self.batch_size):
            batch = prefixed[start : start + self.batch_size]
            tokens = self.tokenizer(batch, context_length=self.max_length).to(self.device)
            with self.torch.inference_mode():
                vectors.append(self.model.encode_text(tokens, normalize=True).detach().float().cpu().numpy())
        overflowed = [count > self.max_length for count in counts]
        return EncodingBatch(
            _normalize(np.vstack(vectors)), counts, list(overflowed),
            chunk_counts=[1] * len(texts), overflowed=overflowed,
        )

    @property
    def supports_images(self) -> bool:
        return True

    def encode_images(self, paths: list[str | Path]) -> np.ndarray:
        from PIL import Image

        vectors: list[np.ndarray] = []
        dtype = next(self.model.parameters()).dtype
        for start in range(0, len(paths), self.batch_size):
            tensors = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device, dtype=dtype)
            with self.torch.inference_mode():
                vectors.append(self.model.encode_image(batch, normalize=True).detach().float().cpu().numpy())
        return _normalize(np.vstack(vectors))


class Mamba3Encoder(TextEncoder):
    """Last-token baseline for the raw Mamba-3 causal language model backbone."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        except ImportError as exc:
            raise RuntimeError(
                "The mamba3 backend needs the latest official mamba_ssm source on Linux: "
                "pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation"
            ) from exc
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = str(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if not self.device.startswith("cuda"):
            raise RuntimeError("The official Mamba-3 Triton kernels require a CUDA device.")
        self.model_id = str(spec["model_id"])
        self.tokenizer_id = str(spec.get("tokenizer_id", "meta-llama/Llama-3.1-8B"))
        self.max_length = int(spec.get("max_length", 2048))
        self.kernel_min_length = int(spec.get("kernel_min_length", 64))
        dtype_name = str(spec.get("dtype", "bfloat16"))
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype_name not in dtype_map:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        self.model = MambaLMHeadModel.from_pretrained(
            self.model_id, device=self.device, dtype=dtype_map[dtype_name]
        ).eval()

    def encode(self, texts: list[str]) -> EncodingBatch:
        prefixed = [self.prefix + text for text in texts]
        embeddings: list[np.ndarray] = []
        counts: list[int] = []
        truncated: list[bool] = []
        # Mamba has no attention mask. Encoding one sequence at a time avoids padding
        # tokens changing the recurrent state used for last-token pooling.
        for text in prefixed:
            raw = self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            count = len(raw)
            tokens = self.tokenizer(
                text, add_special_tokens=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )["input_ids"]
            actual_length = tokens.shape[1]
            if actual_length < self.kernel_min_length:
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id
                if pad_id is None:
                    raise ValueError(f"Tokenizer for {self.tokenizer_id} has no pad or EOS token")
                padding = self.torch.full(
                    (1, self.kernel_min_length - actual_length), pad_id, dtype=tokens.dtype
                )
                tokens = self.torch.cat([tokens, padding], dim=1)
            tokens = tokens.to(self.device)
            with self.torch.inference_mode():
                hidden = self.model.backbone(tokens)
            # Right padding is causally invisible to the final real token.
            embeddings.append(hidden[:, actual_length - 1, :].detach().float().cpu().numpy())
            counts.append(count)
            truncated.append(count > self.max_length)
        return EncodingBatch(
            _normalize(np.vstack(embeddings)), counts, truncated,
            chunk_counts=[1] * len(texts), overflowed=list(truncated),
        )


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
        truncated = [n > max_length for n in counts]
        return EncodingBatch(
            np.asarray(embeddings, dtype=np.float32), counts, truncated,
            chunk_counts=[1] * len(texts), overflowed=list(truncated),
        )


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
    if backend == "jina_clip":
        return JinaClipEncoder(spec)
    if backend == "open_clip":
        return OpenClipEncoder(spec)
    if backend == "mamba3":
        return Mamba3Encoder(spec)
    raise ValueError(
        f"Unknown backend '{backend}'. Expected hash, hf_mean_pool, hf_text_features, "
        "sentence_transformers, jina_clip, open_clip, or mamba3."
    )

