import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from caption_bench.data import Caption, load_variants
from caption_bench.gemini import GeminiClient, generate_variants, load_env, validate_batch


SOURCE = "A woman wearing a red coat and white shoes carries a distinctive green umbrella."
CAPTION = Caption("person::train::1.jpg", "person", "train", "1.jpg", SOURCE)
SPECS = [{"name": "summary_10w", "operation": "summary", "max_words": 10}]


def item(text="Woman in red coat, white shoes, carrying green umbrella."):
    return {"variant_type": "summary_10w", "applicable": True, "text": text,
            "retained_facts": ["red coat", "white shoes", "green umbrella"],
            "omitted_facts": ["distinctive"], "change_note": "Retains clothing and umbrella; omits filler.", "skip_reason": ""}


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    (tmp_path / "dataset.jsonl").write_text("\n".join(json.dumps({"domain": "person", "split": "train",
        "filename": f"{i}.jpg", "caption": SOURCE}) for i in (1, 2)), encoding="utf-8")
    (tmp_path / "prompt.md").write_text("Read the whole source and summarize without adding facts.", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dataset": "dataset.jsonl", "env_file": ".env", "model": "gemini-3.1-flash-lite",
        "prompt_file": "prompt.md", "output_dir": "out", "variants": SPECS,
        "validation_retries": 1, "request_interval_seconds": 0}), encoding="utf-8")
    return path


class FakeClient:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def generate(self, prompt, content, schema):
        self.calls.append(json.loads(content))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return {"variants": [item()]}


def test_summary_uses_facts_from_end_and_is_loadable(config_path):
    client = FakeClient()
    path = generate_variants(config_path, sample="1.jpg", client=client)
    variants = load_variants([path])
    assert len(variants) == 1
    assert "green umbrella" in variants[0].text
    assert client.calls[0]["source_caption"] == SOURCE
    assert variants[0].generator.startswith("gemini:gemini-3.1-flash-lite:")
    assert "retained_facts" in json.loads(path.read_text(encoding="utf-8"))


def test_over_target_is_accepted_unchanged_without_regeneration(config_path):
    long = item("Woman wearing red coat and white shoes carrying a distinctive green umbrella outdoors.")
    client = FakeClient([{"variants": [long]}, {"variants": [item()]}])
    path = generate_variants(config_path, limit=1, client=client)
    assert len(client.calls) == 1
    assert load_variants([path])[0].text == long["text"]
    row = json.loads(path.read_text())
    assert row["word_count"] > row["target_words"] == 10
    assert "max_words" not in client.calls[0]["operations"][0]


def test_failure_saves_previous_caption_and_resume_does_not_repeat(config_path):
    client = FakeClient([{"variants": [item()]}, RuntimeError("temporary API failure")])
    with pytest.raises(RuntimeError):
        generate_variants(config_path, client=client)
    assert len(load_variants([config_path.parent / "out/variants.jsonl"])) == 1
    resumed = FakeClient()
    path = generate_variants(config_path, client=resumed)
    assert len(resumed.calls) == 1
    assert resumed.calls[0]["sample_id"].endswith("2.jpg")
    assert len(load_variants([path])) == 2
    assert len({(v.sample_id, v.variant_type) for v in load_variants([path])}) == 2


def test_resume_rejects_changed_prompt(config_path):
    generate_variants(config_path, limit=1, client=FakeClient())
    (config_path.parent / "prompt.md").write_text("Changed instructions", encoding="utf-8")
    with pytest.raises(ValueError, match="Choose a new output_dir"):
        generate_variants(config_path, client=FakeClient())


def test_invalid_outputs_are_not_published(config_path):
    bad = item()
    bad["text"] = ""
    client = FakeClient([{"variants": [bad]}, {"variants": [bad]}])
    with pytest.raises(RuntimeError, match="Validation failed"):
        generate_variants(config_path, limit=1, client=client)
    assert not (config_path.parent / "out/variants.jsonl").exists()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "empty"])
def test_batch_validation(kind):
    variants = [item()]
    if kind == "missing": variants = []
    if kind == "duplicate": variants *= 2
    if kind == "empty": variants[0]["text"] = ""
    with pytest.raises(ValueError):
        validate_batch({"variants": variants}, CAPTION, SPECS)


def test_not_applicable_negative_is_kept_out_of_scored_variants(config_path):
    config = yaml.safe_load(config_path.read_text())
    config["variants"].append({"name": "llm_attribute_exchange", "operation": "attribute_exchange"})
    config_path.write_text(yaml.safe_dump(config))
    skip = {"variant_type": "llm_attribute_exchange", "applicable": False, "text": "", "retained_facts": [],
            "omitted_facts": [], "change_note": "", "skip_reason": "No appropriate pair."}
    path = generate_variants(config_path, limit=1, client=FakeClient([{"variants": [item(), skip]}]))
    assert len(load_variants([path])) == 1
    manifest = json.loads((path.parent / "generation_manifest.json").read_text())
    assert manifest["skipped"][0]["skip_reason"] == "No appropriate pair."


def test_dry_run_needs_no_key_and_makes_no_api_calls(config_path):
    client = FakeClient()
    preview = generate_variants(config_path, sample="1.jpg", dry_run=True, client=client)
    assert not client.calls
    assert json.loads(preview.read_text())["caption_count"] == 1
    assert not (config_path.parent / "out/variants.jsonl").exists()


def test_missing_key_fails_before_network(config_path):
    with pytest.raises(ValueError, match="GEMINI_API_KEY is empty"):
        generate_variants(config_path, limit=1)


def test_env_preserves_existing_credentials_and_handles_quotes(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    path = tmp_path / ".env"
    path.write_text('GEMINI_API_KEY="file-key"\nGEMINI_MODEL=gemini-3.1-flash-lite # model\nOTHER=ignored\n')
    load_env(path)
    import os
    assert os.environ["GEMINI_API_KEY"] == "environment-key"
    assert os.environ["GEMINI_MODEL"] == "gemini-3.1-flash-lite"


def test_rest_request_schema_key_header_and_thinking_filter(monkeypatch):
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"thought": True, "text": "not JSON reasoning"}, {"text": json.dumps({"variants": [item()]})}]}}]}).encode()
    def open_request(request, timeout):
        requests.append(request)
        return Response()
    monkeypatch.setattr("caption_bench.gemini.urlopen", open_request)
    result = GeminiClient("secret-key", "gemini-3.1-flash-lite", {}).generate("system", "user", {"type": "object"})
    assert result["variants"][0]["text"] == item()["text"]
    request = requests[0]
    assert "secret-key" not in request.full_url
    assert request.get_header("X-goog-api-key") == "secret-key"
    assert json.loads(request.data)["generationConfig"]["responseJsonSchema"] == {"type": "object"}


def test_auth_failure_does_not_expose_key_or_retry(monkeypatch):
    calls = []
    def fail(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 403, "secret-key", {}, None)
    monkeypatch.setattr("caption_bench.gemini.urlopen", fail)
    with pytest.raises(RuntimeError) as exc:
        GeminiClient("secret-key", "gemini-3.1-flash-lite", {}).generate("system", "user", {})
    assert "403" in str(exc.value) and "secret-key" not in str(exc.value)
    assert len(calls) == 1


def test_generated_variants_run_through_existing_encoder_pipeline(config_path):
    from caption_bench.runner import run_experiment
    source = config_path.parent / "dataset.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows.extend({"domain": "vehicle", "split": "train", "filename": f"car{i}.jpg",
                 "caption": f"Blue sedan with silver wheels and a roof rack number {i}."} for i in (1, 2))
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    generated = generate_variants(config_path, sample="1.jpg", client=FakeClient())
    benchmark = config_path.parent / "benchmark.yaml"
    benchmark.write_text(yaml.safe_dump({"dataset": "dataset.jsonl", "output_dir": "run",
        "transforms": [{"name": "identity"}], "variant_files": [str(generated)],
        "models": [{"name": "hash", "backend": "hash", "dimension": 128}]}))
    run_dir = run_experiment(benchmark)
    import csv
    with (run_dir / "details.csv").open(newline="") as f:
        details = list(csv.DictReader(f))
    summary = next(row for row in details if row["variant_type"] == "summary_10w")
    assert summary["variant_text"] == item()["text"]
    assert summary["generator"].startswith("gemini:")
    assert float(summary["cosine"]) > 0
    assert (run_dir / "report.html").exists()
    rows[0]["caption"] += " Added a fact."
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Source caption changed"):
        run_experiment(benchmark)


def test_auxiliary_metadata_does_not_trigger_paid_regeneration(config_path):
    response_item = item()
    response_item.update({"retained_facts": [], "change_note": " ", "skip_reason": "Not applicable"})
    client = FakeClient([{"variants": [response_item]}])
    path = generate_variants(config_path, limit=1, client=client)
    assert len(client.calls) == 1
    row = json.loads(path.read_text())
    assert row["text"] == item()["text"]
    assert row["review_warnings"] == []
    diagnostic = json.loads(next((path.parent / "diagnostics").glob("*-validation-1.json")).read_text())
    assert diagnostic["status"] == "accepted"
    assert diagnostic["response"]["variants"][0]["skip_reason"] == "Not applicable"


def test_empty_caption_still_fails_with_specific_reason():
    response_item = item("")
    with pytest.raises(ValueError, match="text is empty"):
        validate_batch({"variants": [response_item]}, CAPTION, SPECS)


def test_rejected_model_answer_is_saved_without_normalization(config_path):
    empty = item("")
    client = FakeClient([{"variants": [empty]}, {"variants": [item()]}])
    path = generate_variants(config_path, limit=1, client=client)
    diagnostics = [json.loads(p.read_text()) for p in (path.parent / "diagnostics").glob("*.json")]
    rejected = next(d for d in diagnostics if d["status"] == "rejected")
    assert "text is empty" in rejected["reason"]
    assert rejected["response"]["variants"][0]["text"] == ""
    assert "review_warnings" not in rejected["response"]["variants"][0]


def test_http_and_content_retries_share_request_cap(config_path, monkeypatch):
    import io
    calls = []
    long = item("one " * 11)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"candidates": [{"finishReason": "STOP", "content": {
                "parts": [{"text": json.dumps({"variants": []})}]}}]}).encode()
    def fake_request(request, timeout):
        calls.append(request)
        if len(calls) in {1, 3}:
            raise HTTPError(request.full_url, 503, "Unavailable", {}, io.BytesIO(b'{"error":{"message":"Unavailable"}}'))
        return Response()
    monkeypatch.setattr("caption_bench.gemini.urlopen", fake_request)
    monkeypatch.setattr("caption_bench.gemini.time.sleep", lambda _: None)
    client = GeminiClient("secret-key", "gemini-3.1-flash-lite",
                          {"request_retries": 1, "max_api_requests_per_caption": 3},
                          audit_dir=config_path.parent / "api-audit")
    with pytest.raises(RuntimeError, match="limit of 3 API requests"):
        generate_variants(config_path, limit=1, client=client)
    assert len(calls) == 3
    requests = [json.loads(p.read_text()) for p in (config_path.parent / "api-audit").glob("*.json")]
    assert [r["http_status"] for r in sorted(requests, key=lambda r: r["request_number"])] == [503, 200, 503]


def test_error_diagnostics_redact_credentials(tmp_path, monkeypatch):
    import io
    def fake_request(request, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"error":{"message":"Invalid secret-key"}}'))
    monkeypatch.setattr("caption_bench.gemini.urlopen", fake_request)
    client = GeminiClient("secret-key", "gemini-3.1-flash-lite", {}, audit_dir=tmp_path)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        client.generate("system", "content", {})
    log = next(tmp_path.glob("*.json")).read_text()
    assert "secret-key" not in log
    assert "[REDACTED]" in log
    assert "x-goog-api-key" not in log.lower()


def test_metadata_can_be_paraphrased_or_absent_and_summary_can_be_under_target():
    response = {"variants": [{"variant_type": "summary_10w", "text": "Woman holding umbrella.",
                              "omitted_facts": ["Clothing details omitted."]}]}
    result = validate_batch(response, CAPTION, SPECS)[0]
    assert result["word_count"] == 3
    assert result["target_words"] == 10
    assert result["omitted_facts"] == ["Clothing details omitted."]
    assert result["review_warnings"] == []


def test_saved_response_is_reused_without_network_and_preserves_origin(config_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("Saved response recovery must not send an API request")
    monkeypatch.setattr("caption_bench.gemini.urlopen", forbid_network)
    original = {"variants": [{"variant_type": "summary_10w", "text": "one " * 11,
                               "omitted_facts": ["Paraphrased metadata is accepted."]}]}
    diagnostic = config_path.parent / "diagnostics/saved.json"
    diagnostic.parent.mkdir()
    diagnostic.write_text(json.dumps({"sample_id": CAPTION.sample_id, "response": original}))
    api_folder = diagnostic.parent / "api_requests"
    api_folder.mkdir()
    (api_folder / "request.json").write_text(json.dumps({"model": "gemini-3.1-flash-lite",
        "visible_response": json.dumps(original), "request_payload": {
            "systemInstruction": {"parts": [{"text": "old strict prompt"}]},
            "contents": [{"parts": [{"text": json.dumps({"sample_id": CAPTION.sample_id, "source_caption": SOURCE})}]}]}}))
    path = generate_variants(config_path, sample="1.jpg", reuse_response=diagnostic)
    row = json.loads(path.read_text())
    assert row["word_count"] == 11 and row["target_words"] == 10
    assert row["generator"].startswith("gemini:gemini-3.1-flash-lite:recovered-")
    assert row["recovered_from"] == str(diagnostic.resolve())
    next_client = FakeClient()
    generate_variants(config_path, sample="1.jpg", client=next_client)
    assert next_client.calls == []


def test_human_description_has_no_length_target_and_needs_one_request(config_path):
    config = yaml.safe_load(config_path.read_text())
    config["variants"] = [{"name": "llm_human_description", "operation": "human_description"}]
    config_path.write_text(yaml.safe_dump(config))
    text = "She is wearing a red coat and white shoes, and she is holding a green umbrella."
    client = FakeClient([{"variants": [{"variant_type": "llm_human_description", "text": text}]}])
    path = generate_variants(config_path, sample="1.jpg", client=client)
    row = json.loads(path.read_text())
    assert row["text"] == text
    assert row["target_words"] is None
    assert row["word_count"] == len(text.split())
    assert len(client.calls) == 1
    assert client.calls[0]["operations"] == [{"name": "llm_human_description", "operation": "human_description"}]
