"""Tests for privacy-safe local Langfuse candidate-output generation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_langfuse_candidate_outputs.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_langfuse_candidate_outputs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_contract(**overrides):
    base = {
        "dataset_item_id": "item_1",
        "source_trace_id": "trace_1",
        "promotion_reason": "canonical_success",
        "must": ["identify the trace shape", "report credentials only as presence/length booleans"],
        "must_not": ["copy raw trace input/output payloads", "print raw keys or authorization headers"],
        "checks": ["tool_outputs_present", "privacy_findings_zero_or_redacted"],
        "deterministic_checks": [
            {"name": "tool_outputs_present", "type": "deterministic_artifact_check", "target": "execution_artifacts"},
            {"name": "privacy_findings_zero_or_redacted", "type": "secret_scan", "target": "candidate_output"},
        ],
    }
    base.update(overrides)
    return base


def plan(*contracts):
    return {
        "mode": "read_only_eval_plan",
        "dataset_name": "hermes/turn-regression/pilot",
        "contracts": list(contracts),
    }


def test_build_candidate_outputs_summarizes_contract_and_evidence_without_raw_payloads():
    script = load_script()
    artifact_evidence = {
        "trace_1": {
            "tool_outputs_present": True,
            "tool_call_ids_and_args_present": True,
            "summary": {
                "trace_id": "trace_1",
                "tool_observations": 2,
                "tool_output_present_count": 2,
                "tool_null_output_count": 0,
                "tool_call_id_present_count": 2,
                "tool_args_present_count": 2,
            },
        }
    }

    outputs = script.build_candidate_outputs(
        plan(plan_contract()),
        artifact_evidence=artifact_evidence,
        limit=1,
    )
    serialized = json.dumps(outputs, sort_keys=True)

    assert list(outputs) == ["item_1"]
    text = outputs["item_1"]
    assert "dataset_item_id: item_1" in text
    assert "source_trace_id: trace_1" in text
    assert "promotion_reason: canonical_success" in text
    assert "tool_observations: 2" in text
    assert "tool_null_output_count: 0" in text
    assert "must:" in text
    assert "must_not:" in text
    assert "raw trace input/output payloads were not copied" in text
    assert "sk-lf-" not in serialized
    assert "pk-lf-" not in serialized
    assert "Bearer " not in serialized


def test_cli_writes_outputs_and_summary_envelope(tmp_path, capsys):
    script = load_script()
    plan_path = tmp_path / "plan.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "outputs.json"
    plan_path.write_text(json.dumps(plan(plan_contract())))
    evidence_path.write_text(json.dumps({
        "mode": "artifact_evidence_extraction_no_write",
        "artifact_evidence": {
            "trace_1": {
                "tool_outputs_present": True,
                "summary": {"trace_id": "trace_1", "tool_observations": 1},
            }
        },
    }))

    exit_code = script.main([
        "--plan-json", str(plan_path),
        "--artifact-evidence-json", str(evidence_path),
        "--output-json", str(output_path),
        "--limit", "1",
    ])
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_path.read_text())

    assert exit_code == 0
    assert stdout_report == file_report
    assert file_report["mode"] == "local_candidate_outputs_no_write"
    assert file_report["write_enabled"] is False
    assert file_report["summary"] == {
        "contract_count": 1,
        "generated_output_count": 1,
        "raw_payloads_copied": False,
    }
    assert file_report["candidate_outputs"].keys() == {"item_1"}
