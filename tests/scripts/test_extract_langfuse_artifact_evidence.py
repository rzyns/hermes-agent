"""Tests for extracting local eval artifact evidence from Hermes/Langfuse run artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_langfuse_artifact_evidence.py"


def load_script():
    spec = importlib.util.spec_from_file_location("extract_langfuse_artifact_evidence", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_tool_shape_evidence_from_raw_langfuse_trace_without_payload_leakage():
    script = load_script()
    trace = {
        "id": "trace_1",
        "name": "Hermes turn",
        "observations": [
            {"id": "chain_1", "type": "CHAIN", "name": "Hermes turn", "output": None},
            {
                "id": "tool_1",
                "type": "TOOL",
                "name": "Tool: read_file",
                "input": {"path": "/tmp/private-file"},
                "output": {"content": "do not copy raw payload"},
                "metadata": {"tool_call_id": "call_1", "args": {"path": "/tmp/private-file"}},
            },
            {
                "id": "tool_2",
                "type": "TOOL",
                "name": "Tool: terminal",
                "input": {"command": "date"},
                "output": "ok",
                "metadata": {"tool_call_id": "call_2", "args": {"command": "date"}},
            },
        ],
    }

    evidence = script.extract_evidence_from_trace(trace)
    serialized = json.dumps(evidence, sort_keys=True)

    assert evidence == {
        "tool_outputs_present": True,
        "tool_call_ids_and_args_present": True,
        "tool_null_outputs_zero": True,
        "tool_null_outputs_recorded": False,
        "all_sampled_tool_outputs_null": False,
        "tool_correlation_gap_classified": False,
        "profile_success_tool_failure_distinguished": False,
        "evidence_artifact_exists": True,
        "summary": {
            "trace_id": "trace_1",
            "tool_observations": 2,
            "tool_output_present_count": 2,
            "tool_null_output_count": 0,
            "tool_call_id_present_count": 2,
            "tool_args_present_count": 2,
        },
    }
    assert "private-file" not in serialized
    assert "do not copy raw payload" not in serialized


def test_empty_args_metadata_counts_as_args_present_for_zero_arg_tools():
    script = load_script()
    trace = {
        "id": "trace_zero_arg_tool",
        "observations": [
            {
                "id": "tool_zero_args",
                "type": "TOOL",
                "name": "Tool: kanban_show",
                "input": {},
                "output": {"shape": "present"},
                "metadata": {"tool_call_id": "call_1", "args": {}},
            }
        ],
    }

    evidence = script.extract_evidence_from_trace(trace)

    assert evidence["tool_call_ids_and_args_present"] is True
    assert evidence["summary"]["tool_args_present_count"] == 1


def test_cli_extracts_evidence_for_multiple_traces(tmp_path, capsys):
    script = load_script()
    input_path = tmp_path / "traces.json"
    output_path = tmp_path / "evidence.json"
    input_path.write_text(json.dumps({
        "body": {
            "data": [
                {
                    "id": "trace_ok",
                    "observations": [
                        {"id": "tool_ok", "type": "TOOL", "input": {"x": 1}, "output": {"ok": True}, "metadata": {"tool_call_id": "call_ok", "args": {"x": 1}}}
                    ],
                },
                {
                    "id": "trace_null",
                    "observations": [
                        {"id": "tool_null", "type": "TOOL", "input": {"x": 2}, "output": None, "metadata": {"tool_call_id": None}}
                    ],
                },
            ]
        }
    }))

    exit_code = script.main([
        "--input-json", str(input_path),
        "--output-json", str(output_path),
    ])
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_path.read_text())

    assert exit_code == 0
    assert stdout_report == file_report
    assert file_report["mode"] == "artifact_evidence_extraction_no_write"
    assert file_report["write_enabled"] is False
    assert file_report["summary"] == {
        "traces_analyzed": 2,
        "tool_observations": 2,
        "tool_null_outputs": 1,
        "evidence_items": 2,
    }
    assert file_report["artifact_evidence"]["trace_ok"]["tool_outputs_present"] is True
    assert file_report["artifact_evidence"]["trace_ok"]["tool_null_outputs_zero"] is True
    assert file_report["artifact_evidence"]["trace_null"]["tool_outputs_present"] is False
    assert file_report["artifact_evidence"]["trace_null"]["tool_null_outputs_recorded"] is True
