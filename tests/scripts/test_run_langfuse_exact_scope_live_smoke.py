"""Tests for exact-scope LF13/LF8 Langfuse live-smoke guards."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_langfuse_exact_scope_live_smoke.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_langfuse_exact_scope_live_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_case(fixture_id: str, *, expected_label: str = "PASS", expected_score=1):
    return {
        "fixture_id": fixture_id,
        "safe_input_summary": "shape-only safe input",
        "candidate_output_summary": "shape-only candidate output",
        "redacted_minimized_evidence_excerpt": "redacted evidence only",
        "must_cite_evidence_refs": ["evidence:1"],
        "score_target_semantics": {"target_type": "fixture", "target_id": fixture_id},
        "non_authorized_meanings_to_guard": ["blocking gate approval"],
        "expected_label": expected_label,
        "expected_score": expected_score,
        "expected_evaluator_output": {"target": {"target_type": "fixture", "stable_id": fixture_id}},
        "expected_confidence_band": "high",
        "category": "canonical_pass_with_complete_minimized_evidence",
        "manual_label": expected_label,
        "negative_trap_if_any": None,
        "privacy_classification": "minimized_shape_only",
        "raw_payload_available_to_judge": "not_available_to_judge_minimized_artifacts_only",
        "source_family": "lf8_fixture",
        "target_type": "fixture",
        "target_id_or_local_ref": fixture_id,
    }


def candidate_payload(script):
    return {"fixtures": [fixture_case(fixture_id) for fixture_id in script.SELECTED_FIXTURE_IDS]}


def review_payload(script, *, dataset_name=None, max_items=5):
    return {
        "live_smoke_scope_proposal": {
            "max_hosted_dataset_name": dataset_name or script.DATASET_NAME,
            "max_dataset_items": max_items,
            "explicitly_forbidden_even_if_gate_approved": script.FORBIDDEN_ACTIONS,
        }
    }


def write_inputs(tmp_path: Path, script, *, review=None, candidate=None):
    candidate_path = tmp_path / "candidate.json"
    review_path = tmp_path / "review.json"
    candidate_path.write_text(json.dumps(candidate or candidate_payload(script)))
    review_path.write_text(json.dumps(review or review_payload(script)))
    return candidate_path, review_path


def test_preflight_cli_is_no_write_and_preserves_exact_scope(tmp_path, monkeypatch, capsys):
    script = load_script()
    candidate_path, review_path = write_inputs(tmp_path, script)
    output_path = tmp_path / "preflight.json"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")

    exit_code = script.main([
        "--candidate-json", str(candidate_path),
        "--review-json", str(review_path),
        "--run-name", "lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
        "--output-json", str(output_path),
    ])
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_path.read_text())

    assert exit_code == 0
    assert stdout_report == file_report
    assert file_report["mode"] == "preflight_no_write"
    assert file_report["write_enabled"] is False
    assert file_report["langfuse_writes_attempted"] is False
    assert file_report["dataset_name"] == script.DATASET_NAME
    assert file_report["scope"]["selected_item_count"] == 5
    assert file_report["scope"]["local_selected_items_secret_like_hits"] == 0
    assert file_report["next_required_flags_for_live_run"] == ["--write", "--confirm-experiment-write"]


def test_validate_exact_scope_rejects_dataset_broadening(tmp_path):
    script = load_script()
    candidate_path, review_path = write_inputs(tmp_path, script)
    items = script.build_items(candidate_path)
    scope = script.approved_scope(review_path)

    with pytest.raises(script.LiveSmokeError, match="dataset mismatch"):
        script.validate_exact_scope(
            dataset_name="hermes/live-evaluator/broadened",
            run_name="lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
            items=items,
            scope=scope,
        )


def test_validate_exact_scope_rejects_more_than_approved_items(tmp_path):
    script = load_script()
    candidate = candidate_payload(script)
    extra_id = "lf8-02-extra-broadening-case"
    candidate["fixtures"].append(fixture_case(extra_id))
    candidate_path, review_path = write_inputs(tmp_path, script, candidate=candidate)
    items = script.build_items(candidate_path, [*script.SELECTED_FIXTURE_IDS, extra_id])
    scope = script.approved_scope(review_path)

    with pytest.raises(script.LiveSmokeError, match="exceeds approved max"):
        script.validate_exact_scope(
            dataset_name=script.DATASET_NAME,
            run_name="lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
            items=items,
            scope=scope,
        )


def test_build_items_rejects_secret_like_fixture_values(tmp_path):
    script = load_script()
    candidate = candidate_payload(script)
    candidate["fixtures"][0]["candidate_output_summary"] = "token" + "=supersecret"
    candidate_path, review_path = write_inputs(tmp_path, script, candidate=candidate)
    items = script.build_items(candidate_path)
    scope = script.approved_scope(review_path)

    with pytest.raises(script.LiveSmokeError, match="secret scan"):
        script.validate_exact_scope(
            dataset_name=script.DATASET_NAME,
            run_name="lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
            items=items,
            scope=scope,
        )


def test_write_requires_double_confirmation(tmp_path):
    script = load_script()
    candidate_path, review_path = write_inputs(tmp_path, script)

    with pytest.raises(script.LiveSmokeError, match="require both --write and --confirm-experiment-write"):
        script.main([
            "--candidate-json", str(candidate_path),
            "--review-json", str(review_path),
            "--run-name", "lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
            "--write",
        ])


def test_dataset_run_score_zero_is_interpreted_as_item_evaluator_target_mismatch():
    script = load_script()

    interpretation = script.interpret_score_readback(
        dataset_run_score_count=0,
        item_evaluators_used=True,
        run_evaluators_used=False,
    )

    assert interpretation["verdict"] == "expected_target_mismatch_not_failed_persistence"
    assert "trace/observation-targeted" in interpretation["explanation"]
    assert "--trace-id" in interpretation["correct_item_score_readback"]


def test_summarize_readbacks_extracts_dataset_run_items_and_score_counts(tmp_path):
    script = load_script()
    run_readback = {
        "body": {
            "id": "run-1",
            "name": "lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
            "datasetRunItems": [
                {"id": "run-item-1", "traceId": "trace-1"},
                {"id": "run-item-2", "trace": {"id": "trace-2"}},
            ],
        }
    }
    score_readback = {"body": {"data": [{"name": "quality", "dataType": "BOOLEAN", "value": 1}]}}

    assert script.summarize_run_readback(run_readback["body"]) == {
        "type": "dataset_run",
        "dataset_run_id_present": True,
        "dataset_run_id": "run-1",
        "dataset_run_name": "lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
        "dataset_run_item_count": 2,
        "trace_id_count": 2,
        "trace_ids": ["trace-1", "trace-2"],
    }
    assert script.summarize_scores_readback(score_readback["body"]) == {
        "score_count": 1,
        "score_names": {"quality": 1},
        "data_types": {"BOOLEAN": 1},
    }


def test_item_level_score_readback_walks_trace_ids_without_raw_trace_payloads():
    script = load_script()
    calls = []

    def fake_cli(args, env):
        calls.append(args)
        if args[:2] == ["datasets", "get-get-run"]:
            return {
                "id": "run-1",
                "name": args[3],
                "datasetRunItems": [{"traceId": "trace-1"}, {"traceId": "trace-2"}],
            }
        if args[:2] == ["scores", "list"] and "--trace-id" in args:
            trace_id = args[args.index("--trace-id") + 1]
            return {"data": [{"name": "lf8_report_only_semantic_match_v1", "dataType": "BOOLEAN", "value": True, "traceId": trace_id}]}
        if args[:2] == ["scores", "list"] and "--dataset-run-id" in args:
            return {"data": []}
        raise AssertionError(args)

    summary = script.readback_item_level_scores(
        env={"LANGFUSE_HOST": "https://example.test", "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
        dataset_name=script.DATASET_NAME,
        run_name="lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
        cli_runner=fake_cli,
        expected_trace_count=2,
        readback_retry_delay_seconds=0,
    )

    assert summary["trace_id_count"] == 2
    assert summary["total_item_level_score_count"] == 2
    assert summary["score_name_counts"] == {"lf8_report_only_semantic_match_v1": 2}
    assert summary["dataset_run_score_summary"]["score_count"] == 0
    assert summary["score_target_reconciliation"]["verdict"] == "expected_target_mismatch_not_failed_persistence"
    assert summary["raw_payloads_persisted"] is False
    assert all("trace-" not in json.dumps(trace_summary) for trace_summary in summary["trace_score_summaries"])
    assert calls[0][:2] == ["datasets", "get-get-run"]


def test_item_level_score_readback_retries_until_dataset_run_items_are_visible():
    script = load_script()
    dataset_calls = 0

    def fake_cli(args, env):
        nonlocal dataset_calls
        if args[:2] == ["datasets", "get-get-run"]:
            dataset_calls += 1
            if dataset_calls == 1:
                return {"id": "run-1", "name": args[3], "datasetRunItems": []}
            return {"id": "run-1", "name": args[3], "datasetRunItems": [{"traceId": "trace-1"}]}
        if args[:2] == ["scores", "list"] and "--trace-id" in args:
            return {"data": [{"name": "lf8_report_only_label", "dataType": "CATEGORICAL", "value": "PASS"}]}
        if args[:2] == ["scores", "list"] and "--dataset-run-id" in args:
            return {"data": []}
        raise AssertionError(args)

    summary = script.readback_item_level_scores(
        env={"LANGFUSE_HOST": "https://example.test", "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
        dataset_name=script.DATASET_NAME,
        run_name="lf13-live-dev-loop-report-only-smoke-20260513T000000Z",
        cli_runner=fake_cli,
        expected_trace_count=1,
        readback_attempts=3,
        readback_retry_delay_seconds=0,
    )

    assert dataset_calls == 2
    assert summary["run_readback_attempts"] == [
        {"attempt": 1, "dataset_run_id_present": True, "dataset_run_item_count": 0, "trace_id_count": 0},
        {"attempt": 2, "dataset_run_id_present": True, "dataset_run_item_count": 1, "trace_id_count": 1},
    ]
    assert summary["trace_id_count"] == 1
    assert summary["total_item_level_score_count"] == 1
