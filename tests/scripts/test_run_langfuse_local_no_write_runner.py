"""Tests for LF13 local/no-write Langfuse dataset/eval runner slice."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_langfuse_local_no_write_runner.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_langfuse_local_no_write_runner", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_payload() -> dict:
    return {
        "mode": "lf13_explicit_source_seed_v1",
        "write_enabled": False,
        "source_policy": {"allow_broad_trace_filters": False},
        "explicit_sources": {
            "trace_ids": ["02edc631b32cd0378abdcfb26936611a"],
            "candidates": [
                {
                    "dataset_item_id": "lf13-fixture-1",
                    "source_trace_id": "02edc631b32cd0378abdcfb26936611a",
                    "session_id": "20260512_173200_a4e8e5",
                    "turn_id": "turn_c329cf679d8a4591ad16f3ee1be0393f",
                    "promotion_reason": "lf11_reviewed_tool_output_trace",
                    "summary": {
                        "tool_observations": 1,
                        "tool_null_output_count": 0,
                        "tool_outputs_present_count": 1,
                        "tool_call_id_present_count": 1,
                        "tool_args_present_count": 1,
                    },
                    "evidence": {
                        "tool_outputs_present": True,
                        "tool_call_ids_and_args_present": True,
                        "tool_null_outputs_zero": True,
                    },
                    "score_summary": {
                        "name": "lf11_report_only_tool_trace_score_write_success",
                        "value": 1,
                    },
                }
            ],
        },
    }


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_runner_emits_lf13_artifact_shape_without_langfuse_writes(tmp_path):
    script = load_script()
    seed_path = write_json(tmp_path / "seed.json", seed_payload())
    output_dir = tmp_path / "out"

    result = script.run_lf13_local_no_write(seed_path, output_dir)

    expected = {
        "candidate_queue.json",
        "safe_trace_summaries.json",
        "privacy_screen_report.json",
        "dataset_fixture_candidates.jsonl",
        "dry_run_results.json",
        "dry_run_results.md",
        "review_packet.json",
        "review_packet.md",
        "hash_manifest.sha256",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert result["mode"] == "lf13_local_no_write_runner"
    assert result["write_enabled"] is False
    assert result["langfuse_writes_attempted"] is False
    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["privacy_status"] == "pass"

    dry_run = json.loads((output_dir / "dry_run_results.json").read_text())
    assert dry_run["write_enabled"] is False
    assert dry_run["langfuse_writes_attempted"] is False
    assert dry_run["local_score_proposals_not_written"][0]["name"] == "lf11_report_only_tool_trace_score_write_success"
    assert dry_run["blocked_future_approvals"]

    fixture = json.loads((output_dir / "dataset_fixture_candidates.jsonl").read_text().splitlines()[0])
    assert fixture["schema"] == "hermes.eval.fixture.v1"
    assert fixture["provenance"]["source_trace_id"] == "02edc631b32cd0378abdcfb26936611a"
    fixture_text = json.dumps(fixture).lower()
    assert "raw_output" not in fixture_text
    assert "tool_payload" not in fixture_text


def test_explicit_source_restriction_rejects_broad_filters(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["trace_filter"] = {"session_prefix": "202605"}
    seed_path = write_json(tmp_path / "seed.json", payload)

    try:
        script.run_lf13_local_no_write(seed_path, tmp_path / "out")
    except script.LF13NoWriteRunnerError as exc:
        assert "broad trace" in str(exc)
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("broad filters must fail closed")


def test_explicit_source_restriction_rejects_nested_broad_filters(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["candidates"][0]["sessionFilter"] = {"session_prefix": "202605"}
    seed_path = write_json(tmp_path / "seed.json", payload)

    try:
        script.run_lf13_local_no_write(seed_path, tmp_path / "out")
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "broad trace" in message
        assert "explicit_sources.candidates[0].sessionFilter" in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("nested broad filters must fail closed")


def test_privacy_screen_fails_closed_on_raw_payload_or_secret(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["candidates"][0]["raw_output"] = "token" + "=thismustnotpersist"
    seed_path = write_json(tmp_path / "seed.json", payload)

    try:
        script.run_lf13_local_no_write(seed_path, tmp_path / "out")
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "privacy screen failed" in message
        assert "thismustnotpersist" not in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("raw payload and secret-like values must fail closed")


def test_privacy_screen_fails_closed_on_nested_camel_case_raw_payload_keys(tmp_path):
    script = load_script()
    raw_key_variants = (
        "rawOutput",
        "toolPayload",
        "toolOutput",
        "rawMessages",
        "tracePayload",
        "observationPayload",
    )

    for raw_key in raw_key_variants:
        payload = seed_payload()
        payload["explicit_sources"]["candidates"][0]["nested"] = {"deeper": {raw_key: "private payload"}}
        seed_path = write_json(tmp_path / f"{raw_key}.json", payload)

        try:
            script.run_lf13_local_no_write(seed_path, tmp_path / raw_key)
        except script.LF13NoWriteRunnerError as exc:
            message = str(exc)
            assert "privacy screen failed" in message
            assert f"explicit_sources.candidates[0].nested.deeper.{raw_key}" in message
        else:  # pragma: no cover - documents fail-closed expectation
            raise AssertionError(f"{raw_key} must fail closed")


def test_explicit_source_restriction_rejects_broad_filter_key_variants(tmp_path):
    script = load_script()
    broad_key_variants = (
        "traceFilters",
        "sessionFilters",
        "allowBroadTraceFilters",
        "allow-broad-trace-filters",
    )

    for broad_key in broad_key_variants:
        payload = seed_payload()
        payload["explicit_sources"]["candidates"][0]["nested"] = {broad_key: {"session_prefix": "202605"}}
        seed_path = write_json(tmp_path / f"{broad_key}.json", payload)

        try:
            script.run_lf13_local_no_write(seed_path, tmp_path / broad_key)
        except script.LF13NoWriteRunnerError as exc:
            message = str(exc)
            assert "broad trace" in message
            assert f"explicit_sources.candidates[0].nested.{broad_key}" in message
        else:  # pragma: no cover - documents fail-closed expectation
            raise AssertionError(f"{broad_key} must fail closed")


def test_privacy_screen_fails_closed_on_structured_secret_keys(tmp_path):
    script = load_script()
    secret_key_variants = ("api_key", "token", "password", "secret")

    for secret_key in secret_key_variants:
        payload = seed_payload()
        payload["explicit_sources"]["candidates"][0]["nested"] = {secret_key: "private-value-should-not-persist"}
        seed_path = write_json(tmp_path / f"{secret_key}.json", payload)

        try:
            script.run_lf13_local_no_write(seed_path, tmp_path / secret_key)
        except script.LF13NoWriteRunnerError as exc:
            message = str(exc)
            assert "privacy screen failed" in message
            assert f"explicit_sources.candidates[0].nested.{secret_key}" in message
            assert "private-value-should-not-persist" not in message
        else:  # pragma: no cover - documents fail-closed expectation
            raise AssertionError(f"{secret_key} must fail closed")


def test_identifier_fields_reject_unsafe_strings_without_persisting_artifacts(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["trace_ids"] = ["safe-trace-id"]
    payload["explicit_sources"]["candidates"][0]["source_trace_id"] = "private trace id should not persist"
    seed_path = write_json(tmp_path / "seed.json", payload)
    output_dir = tmp_path / "out"

    try:
        script.run_lf13_local_no_write(seed_path, output_dir)
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "identifier field contains unsafe characters" in message
        assert "source_trace_id" in message
        assert "private trace id should not persist" not in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("unsafe identifier strings must fail closed")
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_trace_allowlist_rejects_unsafe_strings_without_echoing_values(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["trace_ids"] = ["private trace id should not persist"]
    seed_path = write_json(tmp_path / "seed.json", payload)
    output_dir = tmp_path / "out"

    try:
        script.run_lf13_local_no_write(seed_path, output_dir)
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "identifier field contains unsafe characters" in message
        assert "explicit_sources.trace_ids[0]" in message
        assert "private trace id should not persist" not in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("unsafe trace allowlist identifiers must fail closed")
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_trace_allowlist_rejects_malformed_non_list(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["trace_ids"] = "02edc631b32cd0378abdcfb26936611a"
    seed_path = write_json(tmp_path / "seed.json", payload)

    try:
        script.run_lf13_local_no_write(seed_path, tmp_path / "out")
    except script.LF13NoWriteRunnerError as exc:
        assert "seed explicit_sources.trace_ids must be a list of strings" in str(exc)
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("malformed trace allowlist must fail closed")


def test_identifier_fields_reject_non_string_values(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["candidates"][0]["source_trace_id"] = True
    seed_path = write_json(tmp_path / "seed.json", payload)

    try:
        script.run_lf13_local_no_write(seed_path, tmp_path / "out")
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "identifier field must be a string" in message
        assert "source_trace_id" in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("non-string identifiers must fail closed")


def test_promotion_reason_rejects_falsy_non_string_without_persisting_artifacts(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["candidates"][0]["promotion_reason"] = False
    seed_path = write_json(tmp_path / "seed.json", payload)
    output_dir = tmp_path / "out"

    try:
        script.run_lf13_local_no_write(seed_path, output_dir)
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "identifier field must be a string" in message
        assert "promotion_reason" in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("falsy non-string promotion_reason must fail closed")
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_summary_counts_reject_strings_without_persisting_artifacts(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["candidates"][0]["summary"]["tool_observations"] = "private count string"
    seed_path = write_json(tmp_path / "seed.json", payload)
    output_dir = tmp_path / "out"

    try:
        script.run_lf13_local_no_write(seed_path, output_dir)
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "summary field must be a non-negative integer" in message
        assert "private count string" not in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("string summary counts must fail closed")
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_generated_privacy_failure_does_not_persist_artifacts(tmp_path):
    script = load_script()
    payload = seed_payload()
    payload["explicit_sources"]["candidates"][0]["score_summary"] = {
        "name": "token=private-value-should-not-persist",
        "value": 1,
    }
    seed_path = write_json(tmp_path / "seed.json", payload)
    output_dir = tmp_path / "out"

    try:
        script.run_lf13_local_no_write(seed_path, output_dir)
    except script.LF13NoWriteRunnerError as exc:
        message = str(exc)
        assert "privacy screen failed" in message
        assert "private-value-should-not-persist" not in message
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("generated privacy failures must fail closed")
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_cli_writes_status_json_and_hash_manifest(tmp_path, capsys):
    script = load_script()
    seed_path = write_json(tmp_path / "seed.json", seed_payload())
    output_dir = tmp_path / "out"

    exit_code = script.main(["--seed-json", str(seed_path), "--output-dir", str(output_dir)])
    stdout_report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout_report["write_enabled"] is False
    assert stdout_report["langfuse_writes_attempted"] is False
    assert (output_dir / "hash_manifest.sha256").exists()
    assert "dataset_fixture_candidates.jsonl" in (output_dir / "hash_manifest.sha256").read_text()
