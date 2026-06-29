"""Tests for no-write local Langfuse dataset eval execution."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_langfuse_dataset_eval.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_langfuse_dataset_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_plan_contract(**overrides):
    base = {
        "dataset_item_id": "hermes-item-1",
        "source_trace_id": "trace_1",
        "status": "ACTIVE",
        "promotion_reason": "privacy_case",
        "must": ["mention reviewed evidence"],
        "must_not": ["include API keys", "include bearer tokens"],
        "checks": ["privacy_safe", "grounded_summary"],
        "deterministic_checks": [
            {"name": "privacy_safe", "type": "secret_scan", "target": "candidate_output"},
            {"name": "grounded_summary", "type": "manual_review", "target": "candidate_output"},
        ],
    }
    base.update(overrides)
    return base


def eval_plan(*contracts):
    return {
        "mode": "read_only_eval_plan",
        "dataset_name": "hermes/turn-regression/pilot",
        "summary": {"dataset_item_count": len(contracts)},
        "proposed_experiment": {
            "write_enabled": False,
            "requires_explicit_future_flags": ["--write", "--confirm-experiment-write"],
            "scoring_policy": "deterministic_and_manual_first",
        },
        "contracts": list(contracts),
    }


def test_build_local_eval_results_secret_scan_and_manual_pending_without_langfuse_writes():
    script = load_script()
    plan = eval_plan(eval_plan_contract())
    candidate_outputs = {"hermes-item-1": "The reply mentions reviewed evidence and no credentials."}

    result = script.build_local_eval_results(plan, candidate_outputs=candidate_outputs)

    assert result["mode"] == "local_eval_no_write"
    assert result["dataset_name"] == "hermes/turn-regression/pilot"
    assert result["write_enabled"] is False
    assert result["requires_explicit_future_flags"] == ["--write", "--confirm-experiment-write"]
    assert result["summary"] == {
        "contract_count": 1,
        "evaluated_count": 1,
        "pass_count": 0,
        "fail_count": 0,
        "pending_manual_count": 1,
        "needs_manual_review_count": 1,
        "missing_output_count": 0,
        "secret_findings": 0,
    }
    assert result["items"] == [
        {
            "dataset_item_id": "hermes-item-1",
            "source_trace_id": "trace_1",
            "status": "needs_manual_review",
            "checks": [
                {"name": "privacy_safe", "type": "secret_scan", "status": "pass", "findings": []},
                {"name": "grounded_summary", "type": "manual_review", "status": "pending", "reason": "manual review required"},
            ],
        }
    ]


def test_secret_scan_failure_redacts_candidate_output():
    script = load_script()
    plan = eval_plan(eval_plan_contract())
    candidate_outputs = {"hermes-item-1": "token=supersecret should not appear"}

    result = script.build_local_eval_results(plan, candidate_outputs=candidate_outputs)
    serialized = json.dumps(result, sort_keys=True)

    assert result["summary"]["fail_count"] == 1
    assert result["summary"]["secret_findings"] == 1
    assert "supersecret" not in serialized
    assert result["items"][0]["checks"][0] == {
        "name": "privacy_safe",
        "type": "secret_scan",
        "status": "fail",
        "findings": [{"pattern": "generic_assignment_secret", "preview": "token[REDACTED] should not appear"}],
    }


def test_cli_reads_plan_and_candidate_outputs_and_writes_no_write_results(tmp_path, capsys):
    script = load_script()
    plan_path = tmp_path / "plan.json"
    outputs_path = tmp_path / "outputs.json"
    result_path = tmp_path / "results.json"
    plan_path.write_text(json.dumps(eval_plan(eval_plan_contract())))
    outputs_path.write_text(json.dumps({"hermes-item-1": "mentions reviewed evidence safely"}))

    exit_code = script.main([
        "--plan-json", str(plan_path),
        "--candidate-outputs-json", str(outputs_path),
        "--output-json", str(result_path),
    ])
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(result_path.read_text())

    assert exit_code == 0
    assert stdout_report == file_report
    assert file_report["mode"] == "local_eval_no_write"
    assert file_report["summary"]["evaluated_count"] == 1


def test_cli_accepts_candidate_outputs_envelope(tmp_path, capsys):
    script = load_script()
    plan_path = tmp_path / "plan.json"
    outputs_path = tmp_path / "candidate-output-envelope.json"
    result_path = tmp_path / "results.json"
    plan_path.write_text(json.dumps(eval_plan(eval_plan_contract())))
    outputs_path.write_text(json.dumps({
        "mode": "local_candidate_outputs_no_write",
        "write_enabled": False,
        "candidate_outputs": {
            "hermes-item-1": "mentions reviewed evidence safely",
        },
    }))

    exit_code = script.main([
        "--plan-json", str(plan_path),
        "--candidate-outputs-json", str(outputs_path),
        "--output-json", str(result_path),
    ])
    file_report = json.loads(result_path.read_text())

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == file_report
    assert file_report["summary"]["missing_output_count"] == 0
    assert file_report["items"][0]["checks"][0]["status"] == "pass"


def test_item_with_passing_automated_checks_and_pending_manual_review_is_not_true_pass():
    script = load_script()
    plan = eval_plan(eval_plan_contract())
    candidate_outputs = {"hermes-item-1": "The reply mentions reviewed evidence and no credentials."}

    result = script.build_local_eval_results(plan, candidate_outputs=candidate_outputs)

    assert result["summary"]["pass_count"] == 0
    assert result["summary"]["needs_manual_review_count"] == 1
    assert result["items"][0]["status"] == "needs_manual_review"


def test_missing_candidate_output_is_not_treated_as_failure():
    script = load_script()
    plan = eval_plan(eval_plan_contract())

    result = script.build_local_eval_results(plan, candidate_outputs={})

    assert result["summary"]["missing_output_count"] == 1
    assert result["summary"]["fail_count"] == 0
    assert result["items"][0]["status"] == "not_run"
    assert result["items"][0]["checks"] == [
        {"name": "privacy_safe", "type": "secret_scan", "status": "not_run", "reason": "candidate output missing"},
        {"name": "grounded_summary", "type": "manual_review", "status": "pending", "reason": "manual review required"},
    ]


def test_cli_runs_artifact_shape_checks_from_local_evidence_without_candidate_output(tmp_path, capsys):
    script = load_script()
    plan_path = tmp_path / "plan.json"
    evidence_path = tmp_path / "artifact-evidence.json"
    result_path = tmp_path / "results.json"
    plan_path.write_text(json.dumps(eval_plan(eval_plan_contract(
        deterministic_checks=[
            {"name": "tool_outputs_present", "type": "deterministic_artifact_check", "target": "execution_artifacts"},
            {"name": "tool_call_ids_and_args_present", "type": "deterministic_artifact_check", "target": "execution_artifacts"},
            {"name": "privacy_safe", "type": "secret_scan", "target": "candidate_output"},
        ],
    ))))
    evidence_path.write_text(json.dumps({
        "hermes-item-1": {
            "tool_outputs_present": True,
            "tool_call_ids_and_args_present": True,
        }
    }))

    exit_code = script.main([
        "--plan-json", str(plan_path),
        "--artifact-evidence-json", str(evidence_path),
        "--output-json", str(result_path),
    ])
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(result_path.read_text())

    assert exit_code == 0
    assert stdout_report == file_report
    assert file_report["summary"] == {
        "contract_count": 1,
        "evaluated_count": 1,
        "pass_count": 1,
        "fail_count": 0,
        "pending_manual_count": 0,
        "needs_manual_review_count": 0,
        "missing_output_count": 1,
        "secret_findings": 0,
    }
    assert file_report["items"] == [
        {
            "dataset_item_id": "hermes-item-1",
            "source_trace_id": "trace_1",
            "status": "pass",
            "checks": [
                {"name": "tool_outputs_present", "type": "deterministic_artifact_check", "status": "pass", "evidence_key": "tool_outputs_present"},
                {"name": "tool_call_ids_and_args_present", "type": "deterministic_artifact_check", "status": "pass", "evidence_key": "tool_call_ids_and_args_present"},
                {"name": "privacy_safe", "type": "secret_scan", "status": "not_run", "reason": "candidate output missing"},
            ],
        }
    ]


def test_cli_accepts_extractor_report_artifact_evidence_envelope(tmp_path, capsys):
    script = load_script()
    plan_path = tmp_path / "plan.json"
    evidence_path = tmp_path / "extractor-report.json"
    result_path = tmp_path / "results.json"
    plan_path.write_text(json.dumps(eval_plan(eval_plan_contract(
        deterministic_checks=[
            {"name": "tool_outputs_present", "type": "deterministic_artifact_check", "target": "execution_artifacts"},
        ],
    ))))
    evidence_path.write_text(json.dumps({
        "mode": "artifact_evidence_extraction_no_write",
        "write_enabled": False,
        "artifact_evidence": {
            "trace_1": {"tool_outputs_present": True},
        },
    }))

    exit_code = script.main([
        "--plan-json", str(plan_path),
        "--artifact-evidence-json", str(evidence_path),
        "--output-json", str(result_path),
    ])
    file_report = json.loads(result_path.read_text())

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == file_report
    assert file_report["summary"]["evaluated_count"] == 1
    assert file_report["items"][0]["checks"][0] == {
        "name": "tool_outputs_present",
        "type": "deterministic_artifact_check",
        "status": "pass",
        "evidence_key": "tool_outputs_present",
    }


def test_artifact_check_reports_partial_tool_id_and_args_coverage_with_counts():
    script = load_script()
    plan = eval_plan(eval_plan_contract(
        deterministic_checks=[
            {"name": "tool_call_ids_and_args_present", "type": "deterministic_artifact_check", "target": "execution_artifacts"},
        ],
    ))

    result = script.build_local_eval_results(
        plan,
        candidate_outputs={},
        artifact_evidence={
            "trace_1": {
                "tool_call_ids_and_args_present": False,
                "summary": {
                    "trace_id": "trace_1",
                    "tool_observations": 47,
                    "tool_call_id_present_count": 47,
                    "tool_args_present_count": 46,
                },
            }
        },
    )

    assert result["summary"]["fail_count"] == 1
    assert result["items"][0]["status"] == "fail"
    assert result["items"][0]["checks"] == [
        {
            "name": "tool_call_ids_and_args_present",
            "type": "deterministic_artifact_check",
            "status": "fail",
            "evidence_key": "tool_call_ids_and_args_present",
            "reason": "tool call id and args presence coverage is incomplete",
            "evidence_summary": {
                "trace_id": "trace_1",
                "tool_observations": 47,
                "tool_call_id_present_count": 47,
                "tool_args_present_count": 46,
            },
        }
    ]


def test_tool_null_outputs_zero_is_not_applicable_when_no_tool_observations():
    script = load_script()
    plan = eval_plan(eval_plan_contract(
        deterministic_checks=[
            {"name": "tool_null_outputs_zero", "type": "deterministic_artifact_check", "target": "execution_artifacts"},
        ],
    ))

    result = script.build_local_eval_results(
        plan,
        candidate_outputs={},
        artifact_evidence={
            "trace_1": {
                "tool_null_outputs_zero": False,
                "summary": {
                    "trace_id": "trace_1",
                    "tool_observations": 0,
                    "tool_null_output_count": 0,
                },
            }
        },
    )

    assert result["summary"] == {
        "contract_count": 1,
        "evaluated_count": 1,
        "pass_count": 0,
        "fail_count": 0,
        "pending_manual_count": 0,
        "needs_manual_review_count": 0,
        "missing_output_count": 1,
        "secret_findings": 0,
    }
    assert result["items"][0]["status"] == "not_run"
    assert result["items"][0]["checks"] == [
        {
            "name": "tool_null_outputs_zero",
            "type": "deterministic_artifact_check",
            "status": "not_applicable",
            "reason": "no tool observations present in artifact evidence",
            "evidence_key": "tool_null_outputs_zero",
            "evidence_summary": {
                "trace_id": "trace_1",
                "tool_observations": 0,
                "tool_null_output_count": 0,
            },
        }
    ]


def test_cli_refuses_write_without_explicit_confirmation(tmp_path):
    script = load_script()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(eval_plan(eval_plan_contract())))

    try:
        script.main(["--plan-json", str(plan_path), "--write"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("--write without --confirm-experiment-write must fail")


def test_cli_write_with_confirmation_invokes_injected_writer(tmp_path, capsys):
    script = load_script()
    calls = []
    plan_path = tmp_path / "plan.json"
    outputs_path = tmp_path / "outputs.json"
    result_path = tmp_path / "results.json"
    plan_path.write_text(json.dumps(eval_plan(eval_plan_contract())))
    outputs_path.write_text(json.dumps({"hermes-item-1": "mentions reviewed evidence safely"}))

    def fake_writer(results, *, plan, env_file, run_name, run_description):
        calls.append({
            "dataset_name": results["dataset_name"],
            "run_name": run_name,
            "run_description": run_description,
            "env_file": env_file,
            "item_count": len(results["items"]),
        })
        return {
            "mode": "langfuse_experiment_write",
            "write_enabled": True,
            "run_name": run_name,
            "dataset_name": results["dataset_name"],
            "created_run_item_count": 1,
            "failed_run_item_count": 0,
            "items": [{"dataset_item_id": "hermes-item-1", "source_trace_id": "trace_1", "write_status": "created"}],
        }

    exit_code = script.main([
        "--plan-json", str(plan_path),
        "--candidate-outputs-json", str(outputs_path),
        "--output-json", str(result_path),
        "--write",
        "--confirm-experiment-write",
        "--run-name", "approved-smoke-run",
        "--run-description", "approved scoped smoke",
    ], writer=fake_writer)
    file_report = json.loads(result_path.read_text())

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == file_report
    assert calls == [{
        "dataset_name": "hermes/turn-regression/pilot",
        "run_name": "approved-smoke-run",
        "run_description": "approved scoped smoke",
        "env_file": None,
        "item_count": 1,
    }]
    assert file_report["write_enabled"] is True
    assert file_report["langfuse_write"]["created_run_item_count"] == 1
