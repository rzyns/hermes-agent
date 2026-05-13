"""Tests for read-only LF8/LF13 review packet builder."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_langfuse_lf8_review_packet.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_langfuse_lf8_review_packet", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lf8_smoke_summary_payload() -> dict:
    return {
        "schema_version": "lf8_exact_scope_smoke_review_summary_v1",
        "verdict": "PASS_WITH_CAVEATS",
        "report_only_non_blocking": True,
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
        "caveats": [
            "Report-only/non-blocking LF8-04 smoke evidence only.",
            "Dataset-run aggregate score is not required or claimed by this summary.",
        ],
        "non_claims": [
            "no blocking gate promotion",
            "no scheduler/deploy/restart authorization",
            "no production trace/session scoring authorization",
            "no broad trace backfill authorization",
            "no run-level aggregate score persistence claim",
        ],
    }


def test_cli_builds_read_only_json_and_markdown_review_packet(tmp_path):
    script = load_script()
    smoke_summary_path = tmp_path / "smoke-summary.json"
    output_json = tmp_path / "packet.json"
    output_md = tmp_path / "packet.md"
    smoke_summary_path.write_text(json.dumps(lf8_smoke_summary_payload()))

    exit_code = script.main([
        "--smoke-summary-json", str(smoke_summary_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    ])
    packet = json.loads(output_json.read_text())
    markdown = output_md.read_text()

    assert exit_code == 0
    assert packet["schema_version"] == "lf8_lf13_review_packet_v1"
    assert packet["verdict"] == "PASS_WITH_CAVEATS"
    assert packet["live_write_performed_by_this_helper"] is False
    assert packet["langfuse_queries_performed_by_this_helper"] is False
    assert packet["gate_recommendation"]["recommended_next_gate"] == "human_review_before_any_automation_or_scope_expansion"
    assert packet["gate_recommendation"]["blocking_gate_authorized"] is False
    assert packet["scope"]["created_exactly_one_dataset_run"] is True
    assert packet["result_support"]["expected_vs_actual_label_matches"] == 5
    assert packet["score_readback"]["dataset_run_aggregate_score_claimed"] is False
    assert any("no blocking gate promotion" == item for item in packet["non_claims"])
    assert packet["reviewed_artifacts"][0]["artifact_sha256"]
    assert str(tmp_path) not in json.dumps(packet)
    assert "PASS_WITH_CAVEATS" in markdown
    assert "No blocking gate promotion" in markdown
    assert str(tmp_path) not in markdown


def test_review_packet_downgrades_when_aggregate_score_is_claimed(tmp_path):
    script = load_script()
    payload = lf8_smoke_summary_payload()
    payload["score_readback"]["dataset_run_aggregate_score_enabled"] = True
    payload["score_readback"]["dataset_run_aggregate_score_claimed"] = True

    packet = script.build_review_packet(
        smoke_summary=payload,
        reviewed_artifacts=[{"role": "smoke_summary", "artifact_name": "summary.json", "artifact_sha256": "0" * 64}],
    )

    assert packet["verdict"] == "NEEDS_REVIEW"


def test_review_packet_whitelists_and_sanitizes_untrusted_summary_fields(tmp_path):
    script = load_script()
    payload = lf8_smoke_summary_payload()
    payload["scope"]["raw_path"] = f"{tmp_path}/raw-private-payload.json"
    payload["result_support"]["secret_note"] = "sk-lf-not-a-real-secret-but-secret-shaped"
    payload["caveats"].append(f"inspect {tmp_path}/raw-private-payload.json with LANGFUSE_SECRET_KEY=sk-lf-not-real")
    payload["non_claims"].append(f"no leak from {tmp_path}/raw-private-payload.json")

    packet = script.build_review_packet(
        smoke_summary=payload,
        reviewed_artifacts=[{"role": "smoke_summary", "artifact_name": "summary.json", "artifact_sha256": "0" * 64}],
    )
    markdown = script.render_markdown(packet)
    serialized = json.dumps(packet) + markdown

    assert "raw_path" not in packet["scope"]
    assert "secret_note" not in packet["result_support"]
    assert str(tmp_path) not in serialized
    assert "sk-lf" not in serialized
    assert "LANGFUSE_SECRET_KEY" not in serialized


def test_review_packet_requires_passing_health_summary_when_supplied():
    script = load_script()
    health_summary = {
        "schema_version": "lf13_langfuse_dev_loop_health_summary_v1",
        "live_write_performed_by_this_helper": False,
        "langfuse_queries_performed_by_this_helper": False,
        "overall_readiness": {
            "manual_dev_loop_health_snapshot_passed": False,
            "blocking_gate_authorized": False,
            "scheduler_or_watchdog_authorized": False,
            "production_score_authorized": False,
        },
    }

    packet = script.build_review_packet(
        smoke_summary=lf8_smoke_summary_payload(),
        health_summary=health_summary,
        reviewed_artifacts=[{"role": "smoke_summary", "artifact_name": "summary.json", "artifact_sha256": "0" * 64}],
    )

    assert packet["verdict"] == "NEEDS_REVIEW"
