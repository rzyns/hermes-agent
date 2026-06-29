"""Tests for LF4 candidate metadata schema/privacy-screen validation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_langfuse_candidate_metadata.py"


def load_script():
    spec = importlib.util.spec_from_file_location("validate_langfuse_candidate_metadata", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_candidate(**overrides):
    candidate = {
        "candidate_id": "lf4-candidate-001",
        "provenance": {
            "source_pool": "kanban_handoffs_and_review_artifacts",
            "source_id": "t_example",
            "source_artifacts": [
                {
                    "path": "/home/openclaw/.hermes/artifacts/hermes-agent/langfuse-quality-next/example.md",
                    "sha256": "0" * 64,
                }
            ],
        },
        "sanitized_summary": "Read-only Kanban handoff summarizing a local test-failure debugging task.",
        "taxonomy": {
            "task_type": "debugging_or_test_failure",
            "tool_profile": "terminal_readonly",
            "privacy_class": "private_sanitized",
            "side_effect_risk": "local_artifact_only",
            "score_reliability": "artifact_evidence_backed",
            "regression_tier": "core_candidate",
        },
        "risk": {
            "side_effect_risk": "local_artifact_only",
            "external_write_approved": False,
            "destructive_or_credential_change": False,
        },
        "expected_behavior": {
            "must": ["reproduce the failure from sanitized artifacts", "cite artifact hashes"],
            "must_not": ["call external write APIs", "print credentials"],
            "success_criteria": ["targeted tests pass", "privacy screen passes"],
        },
        "privacy_notes": {
            "classification": "private_sanitized",
            "rationale": "Only compact handoff text and artifact hashes are retained.",
            "redactions_required": False,
            "raw_payloads_included": False,
            "private_identifiers_included": False,
        },
        "review_evidence": {
            "reviewer_approved": False,
            "proof_artifacts": [
                {
                    "path": "/home/openclaw/.hermes/artifacts/hermes-agent/langfuse-quality-next/example-review.md",
                    "sha256": "1" * 64,
                }
            ],
        },
    }
    candidate.update(overrides)
    return candidate


def test_safe_candidate_passes_with_required_metadata_and_no_write_contract():
    script = load_script()

    report = script.build_validation_report([safe_candidate()])

    assert report["mode"] == "candidate_metadata_privacy_screen_no_write"
    assert report["write_enabled"] is False
    assert report["summary"] == {"candidate_count": 1, "passed": 1, "needs_redaction": 0, "rejected": 0}
    result = report["results"][0]
    assert result["candidate_id"] == "lf4-candidate-001"
    assert result["status"] == "pass"
    assert result["privacy_screen"]["raw_payloads_detected"] is False
    assert result["privacy_screen"]["secret_like_values_detected"] is False
    assert result["privacy_screen"]["private_identifier_markers_detected"] is False
    assert result["errors"] == []


def test_secret_like_values_and_raw_payload_markers_are_rejected():
    script = load_script()
    candidate = safe_candidate(
        candidate_id="lf4-candidate-secret",
        sanitized_summary="Do not persist this raw_trace_input: sk-lf-abc123SECRET and Bearer abcdefghijklmnop",
    )

    report = script.build_validation_report([candidate])

    result = report["results"][0]
    assert result["status"] == "reject"
    assert result["privacy_screen"]["raw_payloads_detected"] is True
    assert result["privacy_screen"]["secret_like_values_detected"] is True
    assert any(finding["kind"] == "raw_payload_marker" for finding in result["findings"])
    assert any(finding["kind"] == "secret_like_value" for finding in result["findings"])
    serialized = json.dumps(report, sort_keys=True)
    assert "sk-lf-abc123SECRET" not in serialized
    assert "Bearer abcdefghijklmnop" not in serialized


def test_sensitive_candidate_without_raw_payloads_needs_redaction_not_promotion():
    script = load_script()
    candidate = safe_candidate(
        candidate_id="lf4-candidate-redaction",
        taxonomy={
            "task_type": "privacy_or_redaction",
            "tool_profile": "file_read",
            "privacy_class": "sensitive_requires_redaction",
            "side_effect_risk": "none",
            "score_reliability": "manual_review_required",
            "regression_tier": "pilot_candidate",
        },
        risk={
            "side_effect_risk": "none",
            "external_write_approved": False,
            "destructive_or_credential_change": False,
        },
        privacy_notes={
            "classification": "sensitive_requires_redaction",
            "rationale": "The source is allowed only after replacing private identifiers with stable placeholders.",
            "redactions_required": True,
            "raw_payloads_included": False,
            "private_identifiers_included": False,
        },
    )

    report = script.build_validation_report([candidate])

    result = report["results"][0]
    assert result["status"] == "needs_redaction"
    assert result["errors"] == []
    assert any(warning["kind"] == "redaction_required" for warning in result["warnings"])
    assert report["summary"] == {"candidate_count": 1, "passed": 0, "needs_redaction": 1, "rejected": 0}


def test_unapproved_external_write_and_blocking_without_review_are_rejected():
    script = load_script()
    candidate = safe_candidate(
        candidate_id="lf4-candidate-blocking-write",
        taxonomy={
            "task_type": "side_effect_approval",
            "tool_profile": "scheduler_or_messaging_mock",
            "privacy_class": "private_sanitized",
            "side_effect_risk": "external_write_requires_approval",
            "score_reliability": "deterministic",
            "regression_tier": "blocking_regression",
        },
        risk={
            "side_effect_risk": "external_write_requires_approval",
            "external_write_approved": False,
            "destructive_or_credential_change": False,
        },
        review_evidence={"reviewer_approved": False, "proof_artifacts": []},
    )

    report = script.build_validation_report([candidate])

    result = report["results"][0]
    assert result["status"] == "reject"
    assert any(error["kind"] == "unapproved_external_write" for error in result["errors"])
    assert any(error["kind"] == "blocking_without_reviewer_approval" for error in result["errors"])
    assert any(error["kind"] == "blocking_without_proof_artifact" for error in result["errors"])


def test_cli_writes_validation_report(tmp_path, capsys):
    script = load_script()
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "report.json"
    candidates_path.write_text(json.dumps({"candidates": [safe_candidate()]}))

    exit_code = script.main([
        "--candidates-json",
        str(candidates_path),
        "--output-json",
        str(output_path),
    ])

    stdout_summary = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_path.read_text())
    assert exit_code == 0
    assert stdout_summary == file_report["summary"]
    assert file_report["write_enabled"] is False
    assert file_report["results"][0]["status"] == "pass"
