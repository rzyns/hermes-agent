"""Tests for the read-only LF8/LF13 digest runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_langfuse_lf8_read_only_digest.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_langfuse_lf8_read_only_digest", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def review_packet_payload() -> dict:
    return {
        "schema_version": "lf8_lf13_review_packet_v1",
        "verdict": "PASS_WITH_CAVEATS",
        "live_write_performed_by_this_helper": False,
        "langfuse_queries_performed_by_this_helper": False,
        "scope": {
            "dataset_name_matches_exact_lf8_scope": True,
            "run_name_matches_report_only_namespace": True,
            "dataset_run_id_present": True,
            "created_exactly_one_dataset_run": True,
        },
        "result_support": {
            "total": 5,
            "expected_vs_actual_label_matches": 5,
            "boolean_evaluator_passes": 5,
            "stop_condition_hit_count": 0,
            "secret_like_hits_in_evidence_summary": 0,
        },
        "score_readback": {
            "trace_id_count": 5,
            "total_item_level_score_count": 10,
            "expected_item_evaluator_count": 5,
            "expected_label_evaluator_count": 5,
            "dataset_run_aggregate_score_enabled": False,
            "dataset_run_aggregate_score_claimed": False,
        },
        "safety_boundary": {
            "non_blocking_report_only": True,
            "production_trace_or_session_score_writes_performed": False,
            "broad_trace_backfill_performed": False,
            "scheduler_deploy_restart_performed": False,
            "blocking_gate_integration_performed": False,
            "raw_private_trace_tool_payloads_included": False,
            "credential_or_env_values_persisted": False,
            "corpus_broadening_performed": False,
        },
        "gate_recommendation": {
            "blocking_gate_authorized": False,
            "scheduler_or_watchdog_authorized": False,
            "production_score_authorized": False,
            "scope_expansion_authorized": False,
            "fixture_expansion_authorized": False,
        },
        "caveats": ["Report-only/non-blocking LF8-04 smoke evidence only."],
    }


def verification_payload() -> dict:
    return {
        "all_checks_passed": True,
        "secret_scan_findings": [],
        "path_scan_findings": [],
    }


def test_cli_emits_digest_from_latest_verified_packet_without_writes_or_queries(tmp_path):
    script = load_script()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    older_packet = artifact_root / "lf8-lf13-review-packet-20260513T000000Z.json"
    latest_packet = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.json"
    latest_verification = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.verification.json"
    output_json = tmp_path / "digest.json"
    output_md = tmp_path / "digest.md"
    state_file = tmp_path / "state.json"
    write_json(older_packet, {**review_packet_payload(), "verdict": "NEEDS_REVIEW"})
    write_json(latest_packet, review_packet_payload())
    write_json(latest_verification, verification_payload())

    exit_code = script.main([
        "--artifact-root", str(artifact_root),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
        "--state-file", str(state_file),
    ])
    digest = json.loads(output_json.read_text())
    markdown = output_md.read_text()
    state = json.loads(state_file.read_text())

    assert exit_code == 0
    assert digest["schema_version"] == "lf8_lf13_read_only_digest_v1"
    assert digest["status"] == "PASS_WITH_CAVEATS"
    assert digest["changed_since_last_digest"] is True
    assert digest["live_write_performed_by_this_helper"] is False
    assert digest["langfuse_queries_performed_by_this_helper"] is False
    assert digest["scheduler_authorized"] is False
    assert digest["blocking_gate_authorized"] is False
    assert digest["source_packet"]["artifact_name"] == latest_packet.name
    assert digest["scope"]["created_exactly_one_dataset_run"] is True
    assert digest["result_support"]["expected_vs_actual_label_matches"] == 5
    assert digest["score_readback"]["dataset_run_aggregate_score_claimed"] is False
    assert "PASS_WITH_CAVEATS" in markdown
    assert "No Langfuse writes or queries" in markdown
    assert str(tmp_path) not in json.dumps(digest)
    assert str(tmp_path) not in markdown
    assert state["last_source_sha256"] == digest["source_packet"]["artifact_sha256"]


def test_digest_fails_closed_when_latest_verification_has_secret_findings(tmp_path):
    script = load_script()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    packet = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.json"
    verification = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.verification.json"
    write_json(packet, review_packet_payload())
    write_json(verification, {**verification_payload(), "secret_scan_findings": ["LANGFUSE_SECRET_KEY"]})

    digest = script.build_digest(artifact_root=artifact_root, state_file=None)

    assert digest["status"] == "NEEDS_REVIEW"
    assert digest["fail_closed_reasons"] == ["source verification contains secret/path scan findings"]
    assert digest["live_write_performed_by_this_helper"] is False
    assert digest["langfuse_queries_performed_by_this_helper"] is False


def test_digest_fails_closed_when_verification_scan_fields_are_missing(tmp_path):
    script = load_script()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    packet = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.json"
    verification = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.verification.json"
    write_json(packet, review_packet_payload())
    write_json(verification, {"all_checks_passed": True})

    digest = script.build_digest(artifact_root=artifact_root, state_file=None)

    assert digest["status"] == "NEEDS_REVIEW"
    assert "source verification missing explicit clean secret/path scan lists" in digest["fail_closed_reasons"]


def test_digest_sanitizes_packet_controlled_caveats(tmp_path):
    script = load_script()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    packet = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.json"
    verification = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.verification.json"
    payload = review_packet_payload()
    payload["caveats"] = [
        f"inspect {tmp_path}/raw-private.json with LANGFUSE_SECRET_KEY=sk-lf-not-real",
        "api" + "_key=super-secret-value Authorization: " + "Bearer " + "abcdefghijklmnopqrstuvwxyz",
        "-----BEGIN " + "PRIVATE KEY-----\nnot-real\n-----END " + "PRIVATE KEY-----",
        "Windows raw file C:" + "\\\\Users\\\\alice smith\\\\Desktop\\\\raw private.json",
        "UNC raw file " + "\\\\\\\\server\\\\shared folder\\\\raw private.json",
        "Linux root file /root/raw private.json and workspace file /workspace/raw private.json",
        "pass" + "word=\"super secret value\" " + "tok" + "en='another secret value'",
    ]
    write_json(packet, payload)
    write_json(verification, verification_payload())

    digest = script.build_digest(artifact_root=artifact_root)
    markdown = script.render_markdown(digest)
    serialized = json.dumps(digest) + markdown

    assert str(tmp_path) not in serialized
    assert "LANGFUSE_SECRET_KEY" not in serialized
    assert "sk-lf" not in serialized
    assert "super-secret-value" not in serialized
    assert "abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "PRIVATE KEY" not in serialized
    assert "C:" not in serialized
    assert "alice smith" not in serialized
    assert "server" not in serialized
    assert "shared folder" not in serialized
    assert "/root/raw" not in serialized
    assert "/workspace/raw" not in serialized
    assert "super secret value" not in serialized
    assert "another secret value" not in serialized


def test_digest_fails_closed_on_exact_scope_and_result_anomalies(tmp_path):
    script = load_script()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    packet = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.json"
    verification = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.verification.json"
    payload = review_packet_payload()
    payload["scope"]["dataset_name_matches_exact_lf8_scope"] = False
    payload["result_support"]["secret_like_hits_in_evidence_summary"] = 1
    payload["safety_boundary"]["non_blocking_report_only"] = False
    write_json(packet, payload)
    write_json(verification, verification_payload())

    digest = script.build_digest(artifact_root=artifact_root, state_file=None)

    assert digest["status"] == "NEEDS_REVIEW"
    assert "exact LF8 dataset scope is not true" in digest["fail_closed_reasons"]
    assert "secret-like evidence hit count is not 0" in digest["fail_closed_reasons"]
    assert "non-blocking report-only boundary is not true" in digest["fail_closed_reasons"]


def test_cli_does_not_advance_state_when_digest_output_write_fails(tmp_path, monkeypatch):
    script = load_script()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    packet = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.json"
    verification = artifact_root / "lf8-lf13-review-packet-20260513T010000Z.verification.json"
    output_json = tmp_path / "digest.json"
    output_md = tmp_path / "digest.md"
    state_file = tmp_path / "state.json"
    write_json(packet, review_packet_payload())
    write_json(verification, verification_payload())

    original_write_json = script.write_json

    def fail_digest_json(path, payload):
        if path == output_json:
            raise OSError("simulated digest write failure")
        original_write_json(path, payload)

    monkeypatch.setattr(script, "write_json", fail_digest_json)

    try:
        script.main([
            "--artifact-root", str(artifact_root),
            "--output-json", str(output_json),
            "--output-md", str(output_md),
            "--state-file", str(state_file),
        ])
    except OSError as exc:
        assert "simulated digest write failure" in str(exc)
    else:  # pragma: no cover - the assertion above should always be reached
        raise AssertionError("expected simulated digest write failure")

    assert not state_file.exists()
