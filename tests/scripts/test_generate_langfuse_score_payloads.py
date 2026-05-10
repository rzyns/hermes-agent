"""Tests for conservative Langfuse score-payload materialization."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_langfuse_score_payloads.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_langfuse_score_payloads", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_eval(**summary_overrides):
    summary = {
        "contract_count": 2,
        "evaluated_count": 2,
        "fail_count": 0,
        "missing_output_count": 0,
        "needs_manual_review_count": 2,
        "pass_count": 0,
        "pending_manual_count": 3,
        "secret_findings": 0,
    }
    summary.update(summary_overrides)
    return {
        "dataset_name": "hermes/turn-regression/pilot",
        "summary": summary,
        "items": [
            {"dataset_item_id": "item-1", "source_trace_id": "trace-1", "status": "needs_manual_review"},
            {"dataset_item_id": "item-2", "source_trace_id": "trace-2", "status": "needs_manual_review"},
        ],
    }


def write_result():
    return {
        "langfuse_write": {
            "dataset_name": "hermes/turn-regression/pilot",
            "run_name": "approved-run",
            "created_run_item_count": 2,
            "failed_run_item_count": 0,
            "items": [
                {"dataset_item_id": "item-1", "source_trace_id": "trace-1", "dataset_run_id": "run-123", "response_id": "run-item-1", "write_status": "created"},
                {"dataset_item_id": "item-2", "source_trace_id": "trace-2", "dataset_run_id": "run-123", "response_id": "run-item-2", "write_status": "created"},
            ],
        }
    }


def semantic_adjudication(**summary_overrides):
    summary = {
        "case_count": 2,
        "manual_check_count": 3,
        "pass_count": 3,
        "fail_count": 0,
        "unclear_count": 0,
        "missing_adjudication_count": 0,
    }
    summary.update(summary_overrides)
    return {
        "mode": "semantic_adjudication_no_write",
        "summary": summary,
        "score_policy_v1": {
            "test_passed": "eligible_for_future_write_gate",
            "task_success": "eligible_for_future_write_gate",
            "reason": "all manual semantic checks adjudicated pass; separate explicit Langfuse score-write approval still required",
        },
    }


def test_build_score_payloads_conservatively_scores_privacy_and_defers_manual():
    script = load_script()

    result = script.build_score_payloads(local_eval(), write_result())

    assert result["mode"] == "langfuse_score_payloads_no_write"
    assert result["write_enabled"] is False
    assert result["dataset_name"] == "hermes/turn-regression/pilot"
    assert result["run_name"] == "approved-run"
    assert result["dataset_run_id"] == "run-123"
    assert result["summary"] == {
        "payload_count": 1,
        "deferred_score_count": 2,
        "manual_pending_count": 3,
        "secret_findings": 0,
        "local_fail_count": 0,
    }
    assert result["score_payloads"] == [
        {
            "endpoint": "/api/public/scores",
            "body": {
                "name": "privacy_safe",
                "value": 1,
                "dataType": "BOOLEAN",
                "datasetRunId": "run-123",
                "comment": "Hermes local evaluator: 0 secret findings across 2 evaluated pilot items. Manual semantic checks remain pending.",
                "metadata": {
                    "dataset_name": "hermes/turn-regression/pilot",
                    "run_name": "approved-run",
                    "producer": "scripts/generate_langfuse_score_payloads.py",
                    "scope": "dataset_run",
                    "manual_pending_count": 3,
                },
            },
        }
    ]
    assert {item["score_name"] for item in result["deferred_scores"]} == {"test_passed", "task_success"}


def test_build_score_payloads_materializes_semantic_scores_after_full_adjudication():
    script = load_script()

    result = script.build_score_payloads(local_eval(pending_manual_count=0, needs_manual_review_count=0), write_result(), semantic_adjudication())

    assert result["summary"] == {
        "payload_count": 3,
        "deferred_score_count": 0,
        "manual_pending_count": 0,
        "secret_findings": 0,
        "local_fail_count": 0,
        "semantic_pass_count": 3,
        "semantic_fail_count": 0,
        "semantic_unclear_count": 0,
    }
    bodies = [payload["body"] for payload in result["score_payloads"]]
    assert [body["name"] for body in bodies] == ["privacy_safe", "test_passed", "task_success"]
    assert all(body["datasetRunId"] == "run-123" for body in bodies)
    assert all(body["dataType"] == "BOOLEAN" and body["value"] == 1 for body in bodies)
    semantic_bodies = {body["name"]: body for body in bodies if body["name"] != "privacy_safe"}
    assert semantic_bodies["test_passed"]["comment"] == "Hermes semantic adjudication: 3/3 manual checks passed with 0 failures and 0 unclear checks."
    assert semantic_bodies["task_success"]["metadata"]["semantic_adjudication_status"] == "fully_passing"
    assert result["deferred_scores"] == []


def test_build_score_payloads_keeps_semantic_scores_deferred_until_fully_passing():
    script = load_script()

    result = script.build_score_payloads(local_eval(), write_result(), semantic_adjudication(unclear_count=1, pass_count=2))

    assert [payload["body"]["name"] for payload in result["score_payloads"]] == ["privacy_safe"]
    assert {item["score_name"] for item in result["deferred_scores"]} == {"test_passed", "task_success"}
    assert all("semantic adjudication is not fully passing" in item["reason"] for item in result["deferred_scores"])


def test_build_score_payloads_refuses_failed_or_ambiguous_write_artifacts():
    script = load_script()
    bad_write = write_result()
    bad_write["langfuse_write"]["items"][1]["dataset_run_id"] = "other-run"

    try:
        script.build_score_payloads(local_eval(fail_count=1), write_result())
    except script.ScorePayloadError as exc:
        assert "local eval has failures" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("failed local eval must not produce score payloads")

    try:
        script.build_score_payloads(local_eval(), bad_write)
    except script.ScorePayloadError as exc:
        assert "exactly one dataset run id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ambiguous dataset run ids must not produce score payloads")


def test_cli_writes_no_write_score_payloads(tmp_path, capsys):
    script = load_script()
    eval_path = tmp_path / "eval.json"
    write_path = tmp_path / "write.json"
    output_path = tmp_path / "scores.json"
    eval_path.write_text(json.dumps(local_eval()))
    write_path.write_text(json.dumps(write_result()))

    exit_code = script.main([
        "--local-eval-json", str(eval_path),
        "--langfuse-write-json", str(write_path),
        "--output-json", str(output_path),
    ])

    file_report = json.loads(output_path.read_text())
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == file_report
    assert file_report["write_enabled"] is False
    assert file_report["summary"]["payload_count"] == 1


def test_cli_writes_no_write_semantic_score_payloads(tmp_path, capsys):
    script = load_script()
    eval_path = tmp_path / "eval.json"
    write_path = tmp_path / "write.json"
    adjudication_path = tmp_path / "adjudication.json"
    output_path = tmp_path / "scores.json"
    eval_path.write_text(json.dumps(local_eval(pending_manual_count=0, needs_manual_review_count=0)))
    write_path.write_text(json.dumps(write_result()))
    adjudication_path.write_text(json.dumps(semantic_adjudication()))

    exit_code = script.main([
        "--local-eval-json", str(eval_path),
        "--langfuse-write-json", str(write_path),
        "--semantic-adjudication-json", str(adjudication_path),
        "--output-json", str(output_path),
    ])

    file_report = json.loads(output_path.read_text())
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == file_report
    assert file_report["write_enabled"] is False
    assert [payload["body"]["name"] for payload in file_report["score_payloads"]] == ["privacy_safe", "test_passed", "task_success"]


def test_cli_can_materialize_only_new_semantic_score_payloads(tmp_path, capsys):
    script = load_script()
    eval_path = tmp_path / "eval.json"
    write_path = tmp_path / "write.json"
    adjudication_path = tmp_path / "adjudication.json"
    output_path = tmp_path / "scores.json"
    eval_path.write_text(json.dumps(local_eval(pending_manual_count=0, needs_manual_review_count=0)))
    write_path.write_text(json.dumps(write_result()))
    adjudication_path.write_text(json.dumps(semantic_adjudication()))

    exit_code = script.main([
        "--local-eval-json", str(eval_path),
        "--langfuse-write-json", str(write_path),
        "--semantic-adjudication-json", str(adjudication_path),
        "--semantic-scores-only",
        "--output-json", str(output_path),
    ])

    file_report = json.loads(output_path.read_text())
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == file_report
    assert file_report["write_enabled"] is False
    assert [payload["body"]["name"] for payload in file_report["score_payloads"]] == ["test_passed", "task_success"]


def test_write_scores_requires_confirmation():
    script = load_script()
    score_payloads = script.build_score_payloads(local_eval(), write_result())

    try:
        script.write_score_payloads(score_payloads, confirm_score_write=False, post_score=lambda body: {"id": "score-1"})
    except script.ScorePayloadError as exc:
        assert "--confirm-score-write" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("score writes must require explicit confirmation")


def test_write_scores_invokes_injected_writer_for_single_privacy_score():
    script = load_script()
    score_payloads = script.build_score_payloads(local_eval(), write_result())
    calls = []

    def fake_post_score(body):
        calls.append(body)
        return {"id": "score-1", "name": body["name"]}

    result = script.write_score_payloads(score_payloads, confirm_score_write=True, post_score=fake_post_score)

    assert calls == [score_payloads["score_payloads"][0]["body"]]
    assert result["mode"] == "langfuse_score_write"
    assert result["write_enabled"] is True
    assert result["created_score_count"] == 1
    assert result["failed_score_count"] == 0
    assert result["scores"] == [
        {
            "name": "privacy_safe",
            "dataset_run_id": "run-123",
            "write_status": "created",
            "response_id": "score-1",
        }
    ]


def test_write_scores_invokes_injected_writer_for_exact_semantic_pair():
    script = load_script()
    score_payloads = script.build_score_payloads(
        local_eval(pending_manual_count=0, needs_manual_review_count=0),
        write_result(),
        semantic_adjudication(),
        include_privacy_score=False,
    )
    calls = []

    def fake_post_score(body):
        calls.append(body)
        return {"id": f"score-{body['name']}", "name": body["name"]}

    result = script.write_score_payloads(score_payloads, confirm_score_write=True, post_score=fake_post_score)

    assert [body["name"] for body in calls] == ["test_passed", "task_success"]
    assert result["created_score_count"] == 2
    assert result["failed_score_count"] == 0
    assert result["scores"] == [
        {
            "name": "test_passed",
            "dataset_run_id": "run-123",
            "write_status": "created",
            "response_id": "score-test_passed",
        },
        {
            "name": "task_success",
            "dataset_run_id": "run-123",
            "write_status": "created",
            "response_id": "score-task_success",
        },
    ]


def test_write_scores_rejects_partial_or_mixed_semantic_payloads():
    script = load_script()
    score_payloads = script.build_score_payloads(
        local_eval(pending_manual_count=0, needs_manual_review_count=0),
        write_result(),
        semantic_adjudication(),
        include_privacy_score=False,
    )
    score_payloads["score_payloads"] = score_payloads["score_payloads"][:1]
    try:
        script.write_score_payloads(score_payloads, confirm_score_write=True, post_score=lambda body: {"id": "score-1"})
    except script.ScorePayloadError as exc:
        assert "expected either the single privacy_safe score or exact semantic score pair" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("partial semantic score set must not be written")


def test_score_payloads_expose_and_enforce_per_score_allowlist():
    script = load_script()

    result = script.build_score_payloads(local_eval(), write_result())

    assert result["score_allowlist"]["privacy_safe"] == {
        "dataType": "BOOLEAN",
        "value": 1,
        "scope": "dataset_run",
        "endpoint": "/api/public/scores",
    }
    body = result["score_payloads"][0]["body"]
    body["dataType"] = "NUMERIC"
    try:
        script.validate_score_payloads_against_allowlist(result)
    except script.ScorePayloadError as exc:
        assert "allowlist mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("score payloads must be checked against per-score allowlist")


def test_verify_score_readback_matches_dataset_run_filtered_scores_and_response_ids():
    script = load_script()
    payloads = script.build_score_payloads(local_eval(), write_result())
    write_report = script.write_score_payloads(
        payloads,
        confirm_score_write=True,
        post_score=lambda body: {"id": "score-privacy-safe"},
    )
    readback = {
        "data": [
            {
                "id": "score-privacy-safe",
                "name": "privacy_safe",
                "value": 1,
                "dataType": "BOOLEAN",
                "datasetRunId": "run-123",
            },
            {
                "id": "unrelated-score",
                "name": "privacy_safe",
                "value": 1,
                "dataType": "BOOLEAN",
                "datasetRunId": "other-run",
            },
        ]
    }

    result = script.verify_score_readback(payloads, write_report, readback)

    assert result["mode"] == "langfuse_score_readback_verification"
    assert result["dataset_run_id"] == "run-123"
    assert result["status"] == "passed"
    assert result["matched_score_count"] == 1
    assert result["scores"] == [
        {
            "name": "privacy_safe",
            "status": "passed",
            "expected_response_id": "score-privacy-safe",
            "matched_response_ids": ["score-privacy-safe"],
            "duplicate_status": "none",
        }
    ]


def test_verify_score_readback_blocks_duplicate_dataset_run_scores():
    script = load_script()
    payloads = script.build_score_payloads(local_eval(), write_result())
    write_report = script.write_score_payloads(
        payloads,
        confirm_score_write=True,
        post_score=lambda body: {"id": "score-privacy-safe"},
    )
    duplicate_readback = {
        "data": [
            {"id": "score-privacy-safe", "name": "privacy_safe", "value": 1, "dataType": "BOOLEAN", "datasetRunId": "run-123"},
            {"id": "score-privacy-safe-dup", "name": "privacy_safe", "value": 1, "dataType": "BOOLEAN", "datasetRunId": "run-123"},
        ]
    }

    result = script.verify_score_readback(payloads, write_report, duplicate_readback)

    assert result["status"] == "failed"
    assert result["duplicate_policy"] == "fail_closed_on_multiple_scores_per_dataset_run_name"
    assert result["scores"][0]["duplicate_status"] == "duplicate_dataset_run_score"
