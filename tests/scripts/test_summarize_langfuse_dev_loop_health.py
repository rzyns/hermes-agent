"""Tests for report-only Langfuse dev-loop health summary helper."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "summarize_langfuse_dev_loop_health.py"


def load_script():
    spec = importlib.util.spec_from_file_location("summarize_langfuse_dev_loop_health", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_smoke_payload() -> dict:
    return {
        "schema_version": "lf13_exact_scope_live_evaluator_smoke_evidence_v1",
        "live_mutation_scope": {
            "dataset_name": "hermes/live-evaluator/lf8-pilot-smoke-20260512",
            "run_name": "lf13-live-dev-loop-report-only-smoke-20260513T122409Z-aggregate",
            "dataset_run_id": "run-123",
            "mutations_performed": ["created_dataset_run"],
            "run_level_aggregate_score_enabled": True,
        },
        "aggregate": {
            "total": 5,
            "expected_vs_actual_label_matches": 5,
            "boolean_evaluator_passes": 5,
            "stop_condition_hits": [],
            "secret_like_hits_in_evidence_summary": 0,
        },
        "item_level_score_readback": {
            "trace_id_count": 5,
            "total_item_level_score_count": 10,
            "score_name_counts": {
                "lf8_report_only_semantic_match_v1": 5,
                "lf8_report_only_label": 5,
            },
            "data_type_counts": {"BOOLEAN": 5, "CATEGORICAL": 5},
            "run_readback_attempts": [
                {"attempt": 1, "dataset_run_id_present": True, "dataset_run_item_count": 0, "trace_id_count": 0},
                {"attempt": 2, "dataset_run_id_present": True, "dataset_run_item_count": 5, "trace_id_count": 5},
            ],
            "dataset_run_score_summary": {
                "score_count": 1,
                "score_names": {"lf8_report_only_all_items_passed_v1": 1},
                "data_types": {"BOOLEAN": 1},
            },
        },
        "safety_boundary": {
            "production_trace_or_session_score_writes_performed": False,
            "broad_trace_backfill_performed": False,
            "scheduler_deploy_restart_performed": False,
            "blocking_gate_integration_performed": False,
            "raw_private_trace_tool_payloads_included": False,
            "credential_or_env_values_persisted": False,
            "corpus_broadening_performed": False,
            "non_blocking_report_only": True,
        },
    }


def aggregate_score_payload(*, name: str = "lf8_report_only_all_items_passed_v1", data_type: str = "BOOLEAN", value: object = 1) -> dict:
    return {
        "schema_version": "lf13_dataset_run_aggregate_score_readback_v1",
        "dataset_run_id": "run-123",
        "score_count": 1,
        "expected_score_name": "lf8_report_only_all_items_passed_v1",
        "scores": [
            {
                "name": name,
                "dataType": data_type,
                "value": value,
                "datasetRunId_present": True,
                "source": "API",
            }
        ],
        "raw_payloads_persisted": False,
    }


def test_summary_keeps_telemetry_item_scores_and_dataset_run_aggregate_separate(tmp_path):
    script = load_script()
    live_path = tmp_path / "live.json"
    score_path = tmp_path / "score.json"
    trace_path = tmp_path / "trace.txt"
    tool_path = tmp_path / "tool.txt"
    out_path = tmp_path / "summary.json"
    live_path.write_text(json.dumps(live_smoke_payload()))
    score_path.write_text(json.dumps(aggregate_score_payload()))
    trace_path.write_text(
        "credentials: {'LANGFUSE_PUBLIC_KEY': {'present': True, 'length': 42}}\n"
        'recent: {"id":"blank","name":"","sessionId":null,"tags":[],"createdAt":"2026-05-13T12:44:54Z"}\n'
        'trace_detail: {"id":"trace-1","name":"Hermes discord turn","sessionId":"session-1","tags":["hermes","langfuse"],"metadata":{"platform":"discord","provider":"openai-codex","model":"gpt-5.5","api_mode":"codex_responses","turn_id":"turn-1"},"observation_types":{"CHAIN":1,"GENERATION":2,"TOOL":1},"tool_observations":1,"tool_null_outputs":0}\n'
    )
    tool_path.write_text(
        "host=https://example.test public_len=42 secret_len=42\n"
        "trace=t1 name='Hermes cli turn' session='s1' created=2026-05-13T12:34:28.083Z observations=3 tool_obs=2 tool_null_outputs=0\n"
        "trace=t2 name='' session=None created=2026-05-13T12:44:19.074Z observations=5 tool_obs=4 tool_null_outputs=0\n"
    )

    exit_code = script.main([
        "--live-smoke-json", str(live_path),
        "--aggregate-score-json", str(score_path),
        "--trace-shape-text", str(trace_path),
        "--tool-output-text", str(tool_path),
        "--output-json", str(out_path),
    ])
    summary = json.loads(out_path.read_text())

    assert exit_code == 0
    assert summary["status"] == "report_only_non_blocking_snapshot"
    assert summary["live_write_performed_by_this_helper"] is False
    assert summary["telemetry_health"]["tool_output_sample"]["total_tool_null_outputs"] == 0
    assert summary["exact_lf8_live_evaluator_smoke"]["item_score_readback"]["total_item_level_score_count"] == 10
    assert summary["dataset_run_aggregate_score"]["expected_boolean_true_present"] is True
    assert summary["overall_readiness"]["manual_dev_loop_health_snapshot_passed"] is True
    assert summary["overall_readiness"]["blocking_gate_authorized"] is False
    assert summary["overall_readiness"]["scheduler_or_watchdog_authorized"] is False


def test_cli_can_emit_lf8_smoke_review_without_aggregate_artifact(tmp_path):
    script = load_script()
    live_path = tmp_path / "live.json"
    out_path = tmp_path / "lf8-review.json"
    payload = live_smoke_payload()
    payload["live_mutation_scope"]["run_name"] = "lf13-live-dev-loop-report-only-smoke-20260513T140319Z"
    payload["live_mutation_scope"]["run_level_aggregate_score_enabled"] = False
    payload["item_level_score_readback"]["dataset_run_score_summary"] = {"score_count": 0, "score_names": {}, "data_types": {}}
    live_path.write_text(json.dumps(payload))

    exit_code = script.main([
        "--live-smoke-json", str(live_path),
        "--output-json", str(out_path),
    ])
    summary = json.loads(out_path.read_text())

    assert exit_code == 0
    assert summary["schema_version"] == "lf8_exact_scope_smoke_review_summary_v1"
    assert summary["verdict"] == "PASS_WITH_CAVEATS"
    assert summary["report_only_non_blocking"] is True
    assert summary["live_write_performed_by_this_helper"] is False
    assert summary["langfuse_queries_performed_by_this_helper"] is False
    assert summary["scope"]["dataset_name_matches_exact_lf8_scope"] is True
    assert summary["scope"]["run_name_matches_report_only_namespace"] is True
    assert summary["scope"]["created_exactly_one_dataset_run"] is True
    assert summary["result_support"]["expected_vs_actual_label_matches"] == 5
    assert summary["result_support"]["boolean_evaluator_passes"] == 5
    assert summary["score_readback"]["total_item_level_score_count"] == 10
    assert summary["score_readback"]["dataset_run_aggregate_score_enabled"] is False
    assert "blocking gate" in " ".join(summary["non_claims"])


def test_lf8_smoke_review_requires_dataset_run_id_for_pass(tmp_path):
    script = load_script()
    payload = live_smoke_payload()
    payload["live_mutation_scope"]["run_level_aggregate_score_enabled"] = False
    payload["live_mutation_scope"]["dataset_run_id"] = None

    summary = script.build_lf8_smoke_review_summary(live_smoke=payload)

    assert summary["scope"]["dataset_run_id_present"] is False
    assert summary["verdict"] == "NEEDS_REVIEW"


def test_boolean_aggregate_reconciliation_requires_expected_name_and_type():
    script = load_script()

    assert script.reconcile_dataset_run_aggregate_score(aggregate_score_payload())["expected_boolean_true_present"] is True
    assert script.reconcile_dataset_run_aggregate_score(aggregate_score_payload(value=True))["expected_boolean_true_present"] is True
    assert script.reconcile_dataset_run_aggregate_score(aggregate_score_payload(name="other"))["expected_boolean_true_present"] is False
    assert script.reconcile_dataset_run_aggregate_score(aggregate_score_payload(data_type="NUMERIC"))["expected_boolean_true_present"] is False
    assert script.reconcile_dataset_run_aggregate_score(aggregate_score_payload(value=0))["expected_boolean_true_present"] is False


def test_summary_redacts_untrusted_trace_tool_and_score_values(tmp_path):
    script = load_script()
    live_path = tmp_path / "live.json"
    score_path = tmp_path / "score.json"
    trace_path = tmp_path / "trace.txt"
    tool_path = tmp_path / "tool.txt"
    out_path = tmp_path / "summary.json"
    live_payload = live_smoke_payload()
    live_payload["live_mutation_scope"]["run_name"] = "secret=do-not-copy"
    live_payload["live_mutation_scope"]["dataset_name"] = "secret=do-not-copy"
    live_payload["live_mutation_scope"]["dataset_run_id"] = "session-secret"
    live_payload["live_mutation_scope"]["mutations_performed"] = ["created_dataset_run", "secret=do-not-copy"]
    live_payload["aggregate"]["stop_condition_hits"] = [{"condition": "secret=do-not-copy"}]
    live_payload["aggregate"]["extra"] = "secret=do-not-copy"
    live_payload["item_level_score_readback"]["score_name_counts"]["secret=do-not-copy"] = 1
    live_payload["item_level_score_readback"]["data_type_counts"]["secret=do-not-copy"] = 1
    live_payload["item_level_score_readback"]["run_readback_attempts"][0]["secret"] = "secret=do-not-copy"
    live_path.write_text(json.dumps(live_payload))
    score_path.write_text(json.dumps(aggregate_score_payload(name="secret=do-not-copy", data_type="secret=do-not-copy", value="secret=do-not-copy")))
    trace_path.write_text(
        'trace_detail: {"id":"trace-secret","name":"secret=do-not-copy","sessionId":"session-secret","tags":["hermes"],"metadata":{"platform":"secret=do-not-copy","provider":"secret=do-not-copy","model":"secret=do-not-copy","api_mode":"secret=do-not-copy","token":"secret=do-not-copy"},"observation_types":{"TOOL":1,"secret=do-not-copy":1},"tool_observations":"secret=do-not-copy","tool_null_outputs":"secret=do-not-copy"}\n'
    )
    tool_path.write_text(
        "trace=trace-secret name='secret=do-not-copy' session='session-secret' created=secret=do-not-copy observations=1 tool_obs=1 tool_null_outputs=0\n"
    )

    script.main([
        "--live-smoke-json", str(live_path),
        "--aggregate-score-json", str(score_path),
        "--trace-shape-text", str(trace_path),
        "--tool-output-text", str(tool_path),
        "--output-json", str(out_path),
    ])
    rendered = out_path.read_text()

    assert "secret=do-not-copy" not in rendered
    assert "session-secret" not in rendered
    assert "trace-secret" not in rendered
    assert str(tmp_path) not in rendered
    summary = json.loads(rendered)
    assert summary["dataset_run_aggregate_score"]["scores"][0]["value_shape"] == "str"
    assert "value" not in summary["dataset_run_aggregate_score"]["scores"][0]
    assert summary["telemetry_health"]["trace_shape"]["discord_trace_detail"]["sessionId_present"] is True


def test_cli_rejects_write_like_flags():
    script = load_script()

    for flag in ["--write", "--confirm-experiment-write", "--enable-run-level-aggregate-score"]:
        try:
            script.main([flag])
        except SystemExit as exc:
            assert exc.code != 0
        else:  # pragma: no cover - explicit fail clarity
            raise AssertionError(f"{flag} should be rejected")
