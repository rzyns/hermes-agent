"""Tests for LF5 no-write LLM-as-judge evaluation scaffold."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_langfuse_llm_judge_eval.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_langfuse_llm_judge_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def source_artifact(tmp_path: Path, name: str = "source.md") -> Path:
    path = tmp_path / name
    path.write_text("sanitized artifact pointer only\n")
    return path


def rubric(tmp_path: Path, *, source_exists: bool = True) -> dict:
    source = source_artifact(tmp_path) if source_exists else tmp_path / "missing.md"
    return {
        "schema": "lf5_00_llm_judge_taxonomy_rubric_packet_v1",
        "policy": {
            "advisory_only": True,
            "non_blocking": True,
            "may_not_write_to_langfuse_without_separate_human_approval": True,
        },
        "target_type_decision": {"write_target_in_lf5": "none_local_json_only"},
        "forbidden_actions": ["Langfuse API calls or writes", "Langfuse dataset/run/score writes"],
        "source_artifacts": [str(source)],
        "judge_prompt_sketch": {
            "system": "Use minimized evidence only.",
            "required_output_shape": {
                "fixture_id": "string",
                "judge_abstained": "boolean",
                "abstain_reason": "string|null",
                "privacy_check": "enum",
                "scores": "object",
                "rationale": "string",
                "non_claims": "array",
            },
        },
        "calibration_fixtures": [
            {
                "id": "positive-case",
                "expected": "PASS from privacy_safe=true, no write authorization",
                "expected_negative": False,
                "purpose": "positive calibration",
                "source": "fixture",
            },
            {
                "id": "expected-negative-case",
                "expected": "EXPECTED_NON_PASS must not become write-ready PASS",
                "expected_negative": True,
                "purpose": "expected-negative preservation",
                "source": "fixture",
            },
        ],
        "scores": [
            {"name": "judge_groundedness", "data_type": "CATEGORICAL", "range_or_categories": ["supported", "partially_supported", "unsupported", "abstain_insufficient_evidence"]},
            {"name": "judge_completeness", "data_type": "CATEGORICAL", "range_or_categories": ["complete", "minor_gap", "major_gap", "abstain_not_applicable"]},
            {"name": "judge_instruction_following", "data_type": "CATEGORICAL", "range_or_categories": ["followed", "minor_deviation", "major_deviation", "abstain_conflicting_or_missing_instructions"]},
            {"name": "judge_tool_use_appropriateness", "data_type": "CATEGORICAL", "range_or_categories": ["appropriate", "underused", "overused_or_unsafe", "abstain_no_tool_context"]},
            {"name": "judge_privacy_risk", "data_type": "CATEGORICAL", "range_or_categories": ["no_risk_observed", "possible_risk_needs_review", "risk_observed", "abstain_raw_payload_not_available"]},
            {"name": "judge_expected_negative_preservation", "data_type": "BOOLEAN", "range_or_categories": [0, 1, "abstain_not_expected_negative_fixture"]},
            {"name": "judge_abstained", "data_type": "BOOLEAN", "range_or_categories": [0, 1]},
        ],
    }


def review() -> dict:
    return {
        "schema": "lf5_01_independent_review_v1",
        "mechanical_verdict": "PASS",
        "substantive_verdict": "PASS",
        "overall_verdict": "PASS",
    }


def test_build_llm_judge_eval_writes_local_schema_and_never_live_writes(tmp_path, monkeypatch):
    script = load_script()
    fake_write_key = "sk" + "-lf-should-not-be-read-or-persisted"
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", fake_write_key)
    rubric_path = write_json(tmp_path / "rubric.json", rubric(tmp_path))
    review_path = write_json(tmp_path / "review.json", review())
    output_dir = tmp_path / "out"

    report = script.build_llm_judge_eval(rubric_path, output_dir, review_json=review_path, run_id="test-run")
    saved = json.loads((output_dir / "test-run.json").read_text())

    assert report["schema"] == "lf5_10_no_write_llm_judge_eval_v1"
    assert report["write_enabled"] is False
    assert report["langfuse_api_calls_attempted"] is False
    assert report["langfuse_writes_attempted"] is False
    assert report["scheduler_mutations_attempted"] is False
    assert saved["aggregate"]["fixture_count"] == 2
    assert saved["aggregate"]["expected_negative_fixture_count"] == 1
    assert saved["secret_scan"]["status"] == "passed"
    assert fake_write_key not in json.dumps(saved, sort_keys=True)
    assert (output_dir / "test-run.md").exists()


def test_expected_negative_preservation_is_separate_from_abstain_and_pass_fail(tmp_path):
    script = load_script()
    rubric_path = write_json(tmp_path / "rubric.json", rubric(tmp_path))
    review_path = write_json(tmp_path / "review.json", review())

    report = script.build_llm_judge_eval(rubric_path, tmp_path / "out", review_json=review_path, run_id="expected-negative")
    expected_negative = [item for item in report["results"] if item["expected_negative"]]

    assert len(expected_negative) == 1
    assert expected_negative[0]["expected_negative_preserved"] is True
    assert expected_negative[0]["scores"]["judge_expected_negative_preservation"] == 1
    assert report["aggregate"]["expected_negative_preserved_count"] == 1
    assert report["aggregate"]["abstain_count"] == 0


def test_missing_source_artifact_abstains_without_coercing_to_failure(tmp_path):
    script = load_script()
    rubric_path = write_json(tmp_path / "rubric.json", rubric(tmp_path, source_exists=False))
    review_path = write_json(tmp_path / "review.json", review())

    report = script.build_llm_judge_eval(rubric_path, tmp_path / "out", review_json=review_path, run_id="abstain")

    assert report["evidence_index"]["missing_source_artifact_count"] == 1
    assert report["aggregate"]["abstain_count"] == 2
    assert {item["abstain_reason"] for item in report["results"]} == {"insufficient_minimized_evidence"}
    assert all(item["scores"]["judge_abstained"] == 1 for item in report["results"])


def test_rubric_that_allows_write_fails_closed_before_output(tmp_path):
    script = load_script()
    unsafe = rubric(tmp_path)
    unsafe["target_type_decision"] = {"write_target_in_lf5": "trace"}
    rubric_path = write_json(tmp_path / "unsafe-rubric.json", unsafe)
    review_path = write_json(tmp_path / "review.json", review())

    try:
        script.build_llm_judge_eval(rubric_path, tmp_path / "out", review_json=review_path, run_id="unsafe")
    except script.LLMJudgeRunnerError as exc:
        assert "none_local_json_only" in str(exc)
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("unsafe write target must fail closed")
    assert not (tmp_path / "out" / "unsafe.json").exists()


def test_raw_payload_key_from_judge_result_is_rejected(tmp_path):
    script = load_script()
    rubric_path = write_json(tmp_path / "rubric.json", rubric(tmp_path))
    review_path = write_json(tmp_path / "review.json", review())

    class UnsafeJudge:
        name = "unsafe"

        def judge(self, fixture, rubric, evidence):
            result = script.DeterministicScaffoldJudge().judge(fixture, rubric, evidence)
            result["candidate_output"] = "raw payload must not persist"
            return result

    try:
        script.build_llm_judge_eval(rubric_path, tmp_path / "out", review_json=review_path, judge=UnsafeJudge(), run_id="unsafe-judge")
    except script.LLMJudgeRunnerError as exc:
        assert "raw payload field" in str(exc)
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("raw-payload-bearing judge output must fail closed")


def test_cli_prints_sanitized_status_not_full_results(tmp_path, capsys):
    script = load_script()
    rubric_path = write_json(tmp_path / "rubric.json", rubric(tmp_path))
    review_path = write_json(tmp_path / "review.json", review())

    exit_code = script.main([
        "--rubric-json", str(rubric_path),
        "--review-json", str(review_path),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "cli-run",
    ])
    stdout_report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout_report["write_enabled"] is False
    assert stdout_report["langfuse_api_calls_attempted"] is False
    assert "results" not in stdout_report
    assert "prompt_packets" not in stdout_report
    assert stdout_report["artifacts"]["json"].endswith("cli-run.json")
