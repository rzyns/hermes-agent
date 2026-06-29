"""Tests for human-digestible Langfuse semantic review packets."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_langfuse_human_review_packet.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_langfuse_human_review_packet", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_review_packet_turns_pending_manual_checks_into_plain_prose_cases():
    script = load_script()
    plan = {
        "dataset_name": "hermes/turn-regression/pilot",
        "contracts": [
            {
                "dataset_item_id": "item-1",
                "source_trace_id": "trace-1",
                "promotion_reason": "privacy_case",
                "must": ["say whether this looks privacy-safe"],
                "must_not": ["include raw prompt text"],
                "deterministic_checks": [
                    {"name": "privacy_safe", "type": "secret_scan", "target": "candidate_output"},
                    {"name": "semantic_privacy_classification", "type": "manual_review", "target": "candidate_output"},
                ],
            }
        ],
    }
    eval_result = {
        "items": [
            {
                "dataset_item_id": "item-1",
                "source_trace_id": "trace-1",
                "status": "needs_manual_review",
                "checks": [
                    {"name": "privacy_safe", "type": "secret_scan", "status": "pass", "findings": []},
                    {"name": "semantic_privacy_classification", "type": "manual_review", "status": "pending"},
                ],
            }
        ]
    }
    candidate_outputs = {
        "mode": "local_candidate_outputs_no_write",
        "candidate_outputs": {
            "item-1": "Candidate says: reviewed minimized evidence only; privacy looks acceptable.",
        },
    }
    evidence = {
        "artifact_evidence": {
            "trace-1": {
                "summary": {
                    "trace_id": "trace-1",
                    "tool_observations": 2,
                    "tool_null_output_count": 0,
                    "tool_call_id_present_count": 2,
                    "tool_args_present_count": 2,
                },
                "tool_null_outputs_zero": True,
            }
        }
    }

    packet = script.build_review_packet(
        plan,
        eval_result=eval_result,
        candidate_outputs=candidate_outputs,
        artifact_evidence=evidence,
        max_candidate_chars=200,
    )

    assert packet["mode"] == "human_semantic_review_packet_no_write"
    assert packet["summary"] == {
        "dataset_name": "hermes/turn-regression/pilot",
        "case_count": 1,
        "pending_manual_check_count": 1,
    }
    case = packet["cases"][0]
    assert case["title"] == "Case 1 — privacy_case / item-1"
    assert case["human_question"] == (
        "Does the candidate output make a reasonable semantic judgment for "
        "`semantic_privacy_classification`, given only the minimized evidence below?"
    )
    assert case["plain_prose_context"].startswith("This case was selected because")
    assert "raw prompt" not in json.dumps(case).lower()
    assert case["candidate_output_preview"].startswith("Candidate says")
    assert case["evidence_summary"]["tool_observations"] == 2
    assert case["requested_human_response"] == ["accept", "reject", "unclear", "notes"]


def test_render_markdown_spoon_feeds_cases_without_raw_payloads():
    script = load_script()
    packet = {
        "summary": {"dataset_name": "hermes/turn-regression/pilot", "case_count": 1, "pending_manual_check_count": 1},
        "cases": [
            {
                "title": "Case 1 — privacy_case / item-1",
                "human_question": "Is this acceptable?",
                "plain_prose_context": "Short context.",
                "candidate_output_preview": "Safe preview.",
                "pending_manual_checks": ["semantic_privacy_classification"],
                "evidence_summary": {"tool_observations": 1, "tool_null_output_count": 0},
                "automated_check_statuses": [{"name": "privacy_safe", "status": "pass"}],
                "requested_human_response": ["accept", "reject", "unclear", "notes"],
            }
        ],
    }

    markdown = script.render_markdown(packet)

    assert "# Langfuse Human Semantic Review Packet" in markdown
    assert "## Case 1 — privacy_case / item-1" in markdown
    assert "**Question:** Is this acceptable?" in markdown
    assert "Please respond with" in markdown
    assert "Safe preview." in markdown
