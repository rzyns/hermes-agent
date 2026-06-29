"""Tests for privacy-preserving Langfuse semantic evidence enrichment."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "enrich_langfuse_semantic_evidence.py"


def load_script():
    spec = importlib.util.spec_from_file_location("enrich_langfuse_semantic_evidence", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace_fixture():
    return {
        "id": "trace-1",
        "name": "Hermes cli turn",
        "environment": "local-hanna",
        "sessionId_present": True,
        "input": None,
        "output": None,
        "observations": [
            {"id": "obs-1", "type": "TOOL", "name": "Tool: read_file", "input": {"shape": "present"}, "output": None, "metadata": {}},
            {"id": "obs-2", "type": "TOOL", "name": "Tool: terminal", "input": {"shape": "present"}, "output": None, "metadata": {}},
        ],
    }


def test_enriches_trace_without_copying_raw_name_or_payload():
    script = load_script()
    trace = trace_fixture()
    trace["input"] = {"prompt": "do not copy this raw prompt"}
    trace["name"] = "User asked secret project name"

    evidence = script.enrich_trace(trace)
    serialized = json.dumps(evidence, sort_keys=True)

    assert evidence["trace_id"] == "trace-1"
    assert evidence["trace_name_mode"] == "other_redacted"
    assert evidence["trace_name_prompt_leakage_risk"] == "unknown_non_static_name"
    assert evidence["environment_profile"] == "hanna"
    assert evidence["trace_input_mode"] == "present_redacted"
    assert evidence["trace_output_mode"] == "null"
    assert evidence["tool_output_mode"] == "all_null"
    assert evidence["tool_id_mode"] == "missing_tool_call_ids"
    assert evidence["observation_id_null_count"] == 0
    assert "do not copy" not in serialized
    assert "secret project" not in serialized


def test_maps_manual_checks_to_resolvable_privacy_preserving_evidence():
    script = load_script()
    trace = trace_fixture()
    trace["name"] = "Hermes cli turn"
    evidence = script.enrich_trace(trace)

    resolutions = script.resolve_checks_from_evidence([
        "prompt_leakage_in_name_false",
        "content_mode_classified",
        "blank_orphan_hanna_classified",
        "hanna_profile_taxonomy_detected",
        "turn_id_present",
    ], evidence)

    by_name = {row["check_name"]: row for row in resolutions}
    assert by_name["prompt_leakage_in_name_false"]["verdict"] == "pass"
    assert by_name["content_mode_classified"]["verdict"] == "pass"
    assert by_name["blank_orphan_hanna_classified"]["verdict"] == "pass"
    assert by_name["hanna_profile_taxonomy_detected"]["verdict"] == "pass"
    assert by_name["turn_id_present"]["verdict"] == "unclear"


def test_cli_writes_enriched_evidence_for_packet_cases(tmp_path, capsys):
    script = load_script()
    trace_path = tmp_path / "traces.json"
    packet_path = tmp_path / "packet.json"
    output_path = tmp_path / "enriched.json"
    trace_path.write_text(json.dumps({"body": {"data": [trace_fixture()]}}))
    packet_path.write_text(json.dumps({"cases": [{"dataset_item_id": "item-1", "source_trace_id": "trace-1", "pending_manual_checks": ["trace_name_mode_static"]}]}))

    exit_code = script.main([
        "--trace-shapes-json", str(trace_path),
        "--review-packet-json", str(packet_path),
        "--output-json", str(output_path),
    ])

    written = json.loads(output_path.read_text())
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == written
    assert written["mode"] == "semantic_evidence_enrichment_no_write"
    assert written["write_enabled"] is False
    assert written["summary"]["case_count"] == 1
    assert written["case_evidence"][0]["source_trace_id"] == "trace-1"
    assert written["case_evidence"][0]["check_resolutions"][0]["check_name"] == "trace_name_mode_static"
