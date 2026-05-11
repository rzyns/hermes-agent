"""Tests for no-write Langfuse semantic adjudication aggregation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_langfuse_semantic_adjudication.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_langfuse_semantic_adjudication", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_packet():
    return {
        "mode": "human_review_packet_no_write",
        "write_enabled": False,
        "cases": [
            {
                "dataset_item_id": "item-1",
                "source_trace_id": "trace-1",
                "title": "Case 1",
                "pending_manual_checks": ["semantic_a", "semantic_b"],
                "candidate_output_preview": "Candidate says A and B with caveats.",
                "evidence_summary": {"tool_observations": 2},
            },
            {
                "dataset_item_id": "item-2",
                "source_trace_id": "trace-2",
                "title": "Case 2",
                "pending_manual_checks": ["semantic_c"],
                "candidate_output_preview": "Candidate cannot prove C.",
                "evidence_summary": {"tool_observations": 0},
            },
        ],
    }


def reviewer_fragment():
    return {
        "reviewer": "reviewer-a",
        "claim_scope": "manual semantic adjudication from minimized review packet",
        "case_adjudications": [
            {
                "dataset_item_id": "item-1",
                "check_adjudications": [
                    {"check_name": "semantic_a", "verdict": "pass", "rationale": "supported by candidate caveat"},
                    {"check_name": "semantic_b", "verdict": "unclear", "rationale": "requires raw trace"},
                ],
            },
            {
                "dataset_item_id": "item-2",
                "check_adjudications": [
                    {"check_name": "semantic_c", "verdict": "fail", "rationale": "candidate admits missing proof"},
                ],
            },
        ],
    }


def test_build_adjudication_summary_counts_verdicts_and_blocks_scores():
    script = load_script()

    result = script.build_adjudication(review_packet(), [reviewer_fragment()])

    assert result["mode"] == "semantic_adjudication_no_write"
    assert result["write_enabled"] is False
    assert result["summary"] == {
        "case_count": 2,
        "manual_check_count": 3,
        "pass_count": 1,
        "fail_count": 1,
        "unclear_count": 1,
        "missing_adjudication_count": 0,
    }
    assert result["score_policy_v1"] == {
        "test_passed": "do_not_write",
        "task_success": "do_not_write",
        "reason": "manual semantic adjudication is not fully passing",
    }


def test_build_adjudication_refuses_missing_or_unknown_checks():
    script = load_script()
    fragment = reviewer_fragment()
    fragment["case_adjudications"][0]["check_adjudications"] = [
        {"check_name": "semantic_a", "verdict": "pass", "rationale": "supported"},
        {"check_name": "not_in_packet", "verdict": "pass", "rationale": "bad"},
    ]

    try:
        script.build_adjudication(review_packet(), [fragment])
    except script.SemanticAdjudicationError as exc:
        assert "unknown manual check" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown checks must be refused")


def test_cli_writes_no_write_adjudication(tmp_path, capsys):
    script = load_script()
    packet_path = tmp_path / "packet.json"
    fragment_path = tmp_path / "fragment.json"
    output_path = tmp_path / "adjudication.json"
    packet_path.write_text(json.dumps(review_packet()))
    fragment_path.write_text(json.dumps(reviewer_fragment()))

    exit_code = script.main([
        "--review-packet-json", str(packet_path),
        "--reviewer-fragment-json", str(fragment_path),
        "--output-json", str(output_path),
    ])

    written = json.loads(output_path.read_text())
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == written
    assert written["write_enabled"] is False
    assert written["summary"]["manual_check_count"] == 3
