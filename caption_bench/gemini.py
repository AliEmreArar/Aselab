"""Grounded caption variants through Gemini REST; resumable, with local validation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

from .data import Caption, load_captions

RELATIONS = {
    "summary": "information_reduced",
    "human_description": "information_reduced",
    "paraphrase": "semantic_preserving",
    "synonym": "semantic_preserving",
    "attribute_order": "semantic_preserving",
    "negation": "semantic_changed",
    "color_change": "semantic_changed",
    "attribute_exchange": "semantic_changed",
}


def load_env(path: Path) -> None:
    """Load the two Gemini settings without overriding the process environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.removeprefix("export ").partition("=")
        if sep and name.strip() in {"GEMINI_API_KEY", "GEMINI_MODEL"}:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            if value:
                os.environ.setdefault(name.strip(), value)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def response_schema(names: list[str]) -> dict:
    properties = {
        "variant_type": {"type": "string", "enum": names},
        "applicable": {"type": "boolean"},
        "text": {"type": "string"},
        "retained_facts": {"type": "array", "items": {"type": "string"}},
        "omitted_facts": {"type": "array", "items": {"type": "string"}},
        "change_note": {"type": "string"},
        "skip_reason": {"type": "string"},
    }
    return {"type": "object", "properties": {"variants": {
        "type": "array", "items": {"type": "object", "properties": properties,
        "required": ["variant_type", "text"], "additionalProperties": False},
    }}, "required": ["variants"], "additionalProperties": False}


class GeminiClient:
    def __init__(self, key: str, model: str, config: dict, *, audit_dir: Path | None = None):
        self.key, self.model, self.config = key, model, config
        self.audit_dir = audit_dir
        self.sample_id = "unspecified"
        self.requests_for_caption = 0
        self.request_cap = int(config.get("max_api_requests_per_caption", 3))
        if self.request_cap < 1:
            raise ValueError("max_api_requests_per_caption must be positive.")

    def begin_caption(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.requests_for_caption = 0

    def _audit(self, path: Path | None, record: dict) -> None:
        if path:
            # Request headers are never logged. Redact the key even if an API
            # error unexpectedly echoes it inside its response message.
            safe = json.dumps(record, ensure_ascii=False).replace(self.key, "[REDACTED]")
            atomic_json(path, json.loads(safe))

    def generate(self, system: str, content: str, schema: dict) -> dict:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": content}]}],
            "generationConfig": {
                "temperature": float(self.config.get("temperature", 1.0)),
                "maxOutputTokens": int(self.config.get("max_output_tokens", 16384)),
                "responseMimeType": "application/json", "responseJsonSchema": schema,
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(self.model, safe='')}:generateContent"
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={
            "Content-Type": "application/json", "x-goog-api-key": self.key,
        }, method="POST")
        retries = int(self.config.get("request_retries", 1))
        for attempt in range(retries + 1):
            if self.requests_for_caption >= self.request_cap:
                raise RuntimeError(f"Stopped: {self.sample_id} reached the limit of {self.request_cap} API requests. HTTP retries and content retries share this limit. Inspect saved diagnostics before rerunning.")
            self.requests_for_caption += 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            audit_path = self.audit_dir / f"{stamp}-{digest(self.sample_id)[:8]}-{self.requests_for_caption}.json" if self.audit_dir else None
            record = {"sample_id": self.sample_id, "model": self.model, "request_number": self.requests_for_caption,
                      "request_limit": self.request_cap, "created_at": stamp, "request_payload": payload, "status": "sending"}
            self._audit(audit_path, record)
            print(f"Gemini API [{self.requests_for_caption}/{self.request_cap}] {self.sample_id}", flush=True)
            try:
                with urlopen(request, timeout=float(self.config.get("timeout_seconds", 120))) as response:
                    body = json.load(response)
                candidates = body.get("candidates", [])
                candidate = candidates[0] if candidates else {}
                text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought"))
                record.update({"status": "response", "http_status": 200, "finish_reason": candidate.get("finishReason"),
                               "usage_metadata": body.get("usageMetadata", {}), "visible_response": text})
                self._audit(audit_path, record)
                print(f"  HTTP 200 | finish={candidate.get('finishReason', 'NO_CANDIDATE')}", flush=True)
                if not candidates or candidates[0].get("finishReason") != "STOP":
                    reason = candidates[0].get("finishReason", "NO_CANDIDATE") if candidates else "NO_CANDIDATE"
                    raise ValueError(f"Gemini did not return a complete caption batch ({reason}).")
                # Thinking parts are not the structured answer.
                result = json.loads(text)
                if not isinstance(result, dict):
                    raise ValueError("Gemini response must be a JSON object.")
                return result
            except HTTPError as exc:
                status = exc.code
                try:
                    error_text = exc.read(8192).decode("utf-8", errors="replace")
                except (AttributeError, OSError):
                    error_text = ""
                record.update({"status": "http_error", "http_status": status, "error_response": error_text})
                self._audit(audit_path, record)
                print(f"  HTTP {status}: Google API error; see saved request diagnostics.", flush=True)
                if status not in {429, 500, 502, 503, 504} or attempt == retries:
                    hint = {400: "Check model support and request settings.", 401: "Check GEMINI_API_KEY.",
                            403: "Check API key access/billing.", 404: "Check GEMINI_MODEL.",
                            429: "Quota/rate limit exceeded; try again later."}.get(status, "Try again later.")
                    raise RuntimeError(f"Gemini HTTP {status}. {hint}") from None
                retry_after = exc.headers.get("Retry-After", "")
                delay = min(float(retry_after), 60.0) if retry_after.isdigit() else min(2 ** attempt, 30)
                time.sleep(delay)
            except (URLError, TimeoutError):
                record.update({"status": "connection_error"})
                self._audit(audit_path, record)
                print("  Connection error; no usable response received.", flush=True)
                if attempt == retries:
                    raise RuntimeError("Gemini connection timed out or failed; saved caption batches can be resumed.") from None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Gemini request failed.")


def validate_batch(body: dict, caption: Caption, specs: list[dict]) -> list[dict]:
    items = body.get("variants")
    if not isinstance(items, list):
        raise ValueError("Response is missing the variants array.")
    expected = {spec["name"]: spec for spec in specs}
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each variant must be an object.")
        name = item.get("variant_type")
        if not isinstance(name, str) or name not in expected or name in seen:
            raise ValueError("Variant names must exactly match the requested names without duplicates.")
        seen.add(name)
        spec = expected[name]
        if not isinstance(item.get("text"), str):
            raise ValueError(f"{name}: text must be a string.")
        text = item["text"].strip()
        item["applicable"] = bool(text) if spec["operation"] in {"summary", "human_description"} else item.get("applicable", bool(text))
        for field in ("change_note", "skip_reason"):
            item[field] = item.get(field, "") if isinstance(item.get(field, ""), str) else ""
        for field in ("retained_facts", "omitted_facts"):
            value = item.get(field, [])
            item[field] = [x.strip() for x in value if isinstance(x, str) and x.strip()] if isinstance(value, list) else []
        item["review_warnings"] = []
        item["word_count"] = len(text.split())
        item["target_words"] = spec.get("target_words", spec.get("max_words")) if spec["operation"] == "summary" else None
        if not item["applicable"] and spec["operation"] not in {"summary", "human_description"}:
            continue
        if not text:
            raise ValueError(f"{name}: text is empty; provide the caption itself.")
        # Word counts are experimental measurements, not acceptance gates.
        # Explanations are optional model notes, not exact-quote assertions.
        item["skip_reason"] = ""
        item["change_note"] = item["change_note"].strip()
        item["text"] = text
    if seen != set(expected):
        raise ValueError("Response must include exactly one item per requested variant.")
    return sorted(items, key=lambda item: list(expected).index(item["variant_type"]))


def generation_request(caption: Caption, specs: list[dict], correction: str | None = None) -> str:
    content = {"sample_id": caption.sample_id, "domain": caption.domain, "source_caption": caption.text,
               "operations": specs}
    if correction:
        content["validation_feedback"] = correction + " Return a corrected complete batch, not just the failed item."
    return json.dumps(content, ensure_ascii=False)


def generate_variants(config_path: str | Path, *, limit: int | None = None,
                      sample: str | None = None, dry_run: bool = False, client=None,
                      reuse_response: str | Path | None = None) -> Path:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    def resolve(value):
        return (config_file.parent / value).resolve()
    specs = [s for s in config["variants"] if s.get("enabled", True)]
    names = [s["name"] for s in specs]
    if not specs or len(set(names)) != len(names) or "identity" in names:
        raise ValueError("Enable unique variant names; identity is reserved for the original.")
    for spec in specs:
        if spec["operation"] not in RELATIONS:
            raise ValueError(f"Unknown operation: {spec['operation']}")
        if spec["operation"] == "summary":
            spec["target_words"] = spec.pop("max_words", spec.get("target_words"))
            if type(spec["target_words"]) is not int or spec["target_words"] <= 0:
                raise ValueError("Summary target_words must be a positive approximate length.")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive.")
    load_env(resolve(config.get("env_file", "../.env")))
    model = os.environ.get("GEMINI_MODEL") or config.get("model", "gemini-3.1-flash-lite")
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
        raise ValueError("GEMINI_MODEL must be a model identifier, e.g. gemini-3.1-flash-lite.")
    dataset = resolve(config["dataset"])
    captions = load_captions(dataset)
    if sample:
        captions = [c for c in captions if c.sample_id == sample or c.filename == sample]
        if not captions:
            raise ValueError(f"Sample not found: {sample}")
    if limit:
        captions = captions[:limit]
    recovered_origin = None
    if reuse_response:
        if len(captions) != 1:
            raise ValueError("--reuse-response requires selecting one caption with --sample.")
        saved_path = Path(reuse_response).resolve()
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved.get("sample_id") != captions[0].sample_id or not isinstance(saved.get("response"), dict):
            raise ValueError("Saved response does not belong to the selected caption.")
        matching = None
        for request_path in sorted((saved_path.parent / "api_requests").glob("*.json"), reverse=True):
            request_record = json.loads(request_path.read_text(encoding="utf-8"))
            try:
                parsed_response = json.loads(request_record.get("visible_response", ""))
                source_request = json.loads(request_record["request_payload"]["contents"][0]["parts"][0]["text"])
            except (ValueError, KeyError, IndexError):
                continue
            if parsed_response == saved["response"] and request_record.get("model") == model and source_request.get("sample_id") == captions[0].sample_id and source_request.get("source_caption") == captions[0].text:
                matching = request_record
                break
        if matching is None:
            raise ValueError("Cannot verify source caption/model of the saved response; no API calls were made.")
        original_prompt = matching["request_payload"].get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        recovered_origin = {"recovered_from": str(saved_path), "generation_prompt_sha256": digest(original_prompt),
                            "generator": f"gemini:{model}:recovered-{digest(json.dumps(saved['response'], sort_keys=True))[:12]}"}
        class SavedResponseClient:
            def generate(self, *args):
                return json.loads(json.dumps(saved["response"]))
        client = SavedResponseClient()
        print(f"Reusing saved Gemini response for {captions[0].filename}; no API requests.")
    prompt = resolve(config["prompt_file"]).read_text(encoding="utf-8")
    output_dir = resolve(config["output_dir"])
    output = output_dir / "variants.jsonl"
    signature = digest(json.dumps({"model": model, "prompt": prompt, "specs": specs,
        "dataset": digest(dataset.read_text(encoding="utf-8")),
        "temperature": config.get("temperature", 1.0), "max_output_tokens": config.get("max_output_tokens", 16384)}, sort_keys=True))
    if dry_run:
        preview = output_dir / "request_preview.json"
        atomic_json(preview, {"model": model, "caption_count": len(captions), "system_instruction": prompt,
            "first_request": json.loads(generation_request(captions[0], specs)), "response_schema": response_schema(names)})
        print(f"Dry run: {len(captions)} captions, {len(specs)} variants each; no API calls. Preview: {preview}")
        return preview
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise ValueError("Model, prompt, tasks or dataset changed. Choose a new output_dir to preserve the previous generation.")
    else:
        if output.exists():
            raise ValueError("Existing variant output has no matching checkpoint. Choose a new output_dir.")
        checkpoint = {"signature": signature, "model": model, "created_at": datetime.now(timezone.utc).isoformat(), "captions": {}}
    todo = [c for c in captions if c.sample_id not in checkpoint["captions"]]
    if todo and client is None:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError(f"GEMINI_API_KEY is empty. Enter your key in {resolve(config.get('env_file', '../.env'))}.")
        client = GeminiClient(key, model, config, audit_dir=output_dir / "diagnostics" / "api_requests")
    def publish():
        rows = []
        skipped = []
        for batch in checkpoint["captions"].values():
            for item in batch["variants"]:
                if not item["applicable"]:
                    skipped.append({"sample_id": batch["sample_id"], **item})
                    continue
                spec = next(s for s in specs if s["name"] == item["variant_type"])
                rows.append({"sample_id": batch["sample_id"], "variant_type": item["variant_type"],
                    "text": item["text"], "generator": batch.get("generator", f"gemini:{model}:{signature[:12]}"),
                    "relation": RELATIONS[spec["operation"]], "source_sha256": batch["source_sha256"],
                    "retained_facts": item["retained_facts"], "omitted_facts": item["omitted_facts"],
                    "change_note": item["change_note"], "review_warnings": item.get("review_warnings", []),
                    "word_count": item.get("word_count", len(item["text"].split())), "target_words": item.get("target_words"),
                    **{k: batch[k] for k in ("recovered_from", "generation_prompt_sha256") if k in batch},
                    "generated_at": batch["generated_at"]})
        output_dir.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".jsonl.tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temp.replace(output)
        atomic_json(output_dir / "generation_manifest.json", {"signature": signature, "model": model,
            "prompt_sha256": digest(prompt), "created_at": checkpoint["created_at"],
            "completed_captions": len(checkpoint["captions"]), "variant_count": len(rows), "specs": specs,
            "skipped": skipped, "review_note": "Word counts are measurements only. Optional model notes do not determine acceptance."})
    # Recover the exported JSONL if interruption happened after saving a checkpoint.
    if checkpoint["captions"]:
        publish()
    write_diagnostic = getattr(client, "_audit", atomic_json)
    for index, caption in enumerate(todo):
        if hasattr(client, "begin_caption"):
            client.begin_caption(caption.sample_id)
        correction = None
        retries = int(config.get("validation_retries", 1))
        for attempt in range(retries + 1):
            response = None
            snapshot = None
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            diagnostic_path = output_dir / "diagnostics" / f"{stamp}-{digest(caption.sample_id)[:8]}-validation-{attempt + 1}.json"
            try:
                response = client.generate(prompt, generation_request(caption, specs, correction), response_schema(names))
                # Snapshot the model's visible answer BEFORE normalization.
                snapshot = json.loads(json.dumps(response))
                write_diagnostic(diagnostic_path, {"sample_id": caption.sample_id, "validation_attempt": attempt + 1,
                    "status": "received", "response": snapshot})
                items = validate_batch(response, caption, specs)
                diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                diagnostic.update({"status": "accepted", "review_warnings": {x["variant_type"]: x.get("review_warnings", []) for x in items}})
                write_diagnostic(diagnostic_path, diagnostic)
                break
            except ValueError as exc:
                correction = str(exc)
                write_diagnostic(diagnostic_path, {"sample_id": caption.sample_id, "validation_attempt": attempt + 1,
                    "status": "rejected", "reason": correction, "response": snapshot})
                print(f"  Local validation rejected the response: {correction}", flush=True)
                print(f"  Diagnostics: {diagnostic_path}", flush=True)
                if attempt < retries:
                    print("  Retrying with validation feedback (another API request).", flush=True)
        else:
            saved = len(checkpoint["captions"])
            raise RuntimeError(f"Validation failed for {caption.filename}: {correction} Saved caption batches: {saved}; this caption was not exported. Inspect {diagnostic_path} before retrying.")
        checkpoint["captions"][caption.sample_id] = {"sample_id": caption.sample_id,
            "source_sha256": digest(caption.text), "generated_at": datetime.now(timezone.utc).isoformat(), "variants": items,
            **(recovered_origin or {})}
        atomic_json(checkpoint_path, checkpoint)
        publish()
        print(f"[{index + 1}/{len(todo)}] {caption.filename}: {sum(x['applicable'] for x in items)} variants saved", flush=True)
        if index + 1 < len(todo):
            time.sleep(max(0, float(config.get("request_interval_seconds", 1))))
    print(f"Ready: {len(checkpoint['captions'])} captions generated with {model}. Output: {output}")
    return output
