"""Tests for final targeted Langfuse semantic evidence v2 aggregation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_langfuse_targeted_evidence_v2.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_langfuse_targeted_evidence_v2", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_case(item_id: str, trace_id: str, title: str, promotion_reason: str, pending: list[str], statuses: list[dict] | None = None):
    return {
        "dataset_item_id": item_id,
        "source_trace_id": trace_id,
        "title": title,
        "promotion_reason": promotion_reason,
        "pending_manual_checks": pending,
        "candidate_output_preview": "bounded preview with no secrets",
        "automated_check_statuses": statuses or [],
    }


def test_resolves_credential_checks_from_case_secret_scan_without_copying_candidate_text():
    script = load_script()
    case = base_case(
        "item-privacy",
        "trace-privacy",
        "Case 11 — privacy_case / item-privacy",
        "privacy_case",
        ["credential_values_not_printed"],
        [{"name": "privacy_findings_zero", "status": "pass", "type": "secret_scan"}],
    )

    rows = script.resolve_targeted_checks(case, {}, [case])

    assert rows == [{
        "check_name": "credential_values_not_printed",
        "verdict": "pass",
        "evidence_kind": "secret_scan_status",
        "rationale": "case-level secret scan passed and candidate preview has no known secret-token pattern; raw credential values were not persisted",
    }]
    assert "bounded preview" not in json.dumps(rows)


def test_resolves_comparator_checks_from_dataset_level_class_links():
    script = load_script()
    failure = base_case("item-fail", "trace-fail", "Case 1 — failure / item-fail", "failure", ["contrasted_with_canonical_drugi_trace"])
    canonical = base_case("item-canon", "trace-canon", "Case 4 — canonical_success / item-canon", "canonical_success", ["positive_comparator_recorded"])
    enriched_by_trace = {
        "trace-fail": {"environment_profile": "drugi", "tool_output_mode": "all_null"},
        "trace-canon": {"environment_profile": "drugi", "tool_output_mode": "all_present", "tool_id_mode": "all_tool_call_ids_present"},
    }

    failure_rows = script.resolve_targeted_checks(failure, enriched_by_trace, [failure, canonical])
    canonical_rows = script.resolve_targeted_checks(canonical, enriched_by_trace, [failure, canonical])

    assert failure_rows[0]["verdict"] == "pass"
    assert failure_rows[0]["evidence_kind"] == "comparator_link_shape"
    assert canonical_rows[0]["verdict"] == "pass"
    assert canonical_rows[0]["evidence_kind"] == "comparator_link_shape"


def test_resolves_turn_and_root_context_from_whitelisted_live_metadata_shape():
    script = load_script()
    case = base_case(
        "item-meta",
        "trace-meta",
        "Case 4 — canonical_success / item-meta",
        "canonical_success",
        ["turn_id_present", "root_context_absence_recorded"],
    )
    enriched_by_trace = {
        "trace-meta": {
            "live_metadata_shape_v2": {
                "turn_id_present": True,
                "root_context_keys_present": False,
                "root_context_absence_recorded": True,
            }
        }
    }

    rows = script.resolve_targeted_checks(case, enriched_by_trace, [case])
    by_name = {row["check_name"]: row for row in rows}

    assert by_name["turn_id_present"]["verdict"] == "pass"
    assert by_name["root_context_absence_recorded"]["verdict"] == "pass"


def test_cli_builds_updated_fragment(tmp_path, capsys):
    script = load_script()
    packet_path = tmp_path / "packet.json"
    adjudication_path = tmp_path / "adjudication.json"
    enriched_path = tmp_path / "enriched.json"
    output_path = tmp_path / "fragment.json"

    cases = [
        base_case("item-privacy", "trace-privacy", "Case 11 — privacy_case / item-privacy", "privacy_case", ["credential_values_not_printed"], [{"name": "privacy_findings_zero", "status": "pass", "type": "secret_scan"}]),
    ]
    packet_path.write_text(json.dumps({"cases": cases}))
    adjudication_path.write_text(json.dumps({"check_adjudications": [{"dataset_item_id": "item-privacy", "source_trace_id": "trace-privacy", "title": cases[0]["title"], "check_name": "credential_values_not_printed", "verdict": "unclear", "rationale": "old"}]}))
    enriched_path.write_text(json.dumps({"case_evidence": [{"source_trace_id": "trace-privacy", "enriched_evidence": {}}]}))

    exit_code = script.main([
        "--review-packet-json", str(packet_path),
        "--current-adjudication-json", str(adjudication_path),
        "--enriched-evidence-json", str(enriched_path),
        "--output-json", str(output_path),
    ])

    written = json.loads(output_path.read_text())
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == written
    assert written["summary"]["counts"]["pass"] == 1
    assert written["summary"]["targeted_updates"]["unclear_to_pass"] == 1
    assert written["case_adjudications"][0]["check_adjudications"][0]["verdict"] == "pass"
