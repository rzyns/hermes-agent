#!/usr/bin/env python3
"""Aggregate no-write semantic adjudications for Langfuse review packets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class SemanticAdjudicationError(ValueError):
    """Raised when semantic adjudication input is incomplete or unsafe."""


_VALID_VERDICTS = {"pass", "fail", "unclear"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _packet_checks(review_packet: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    checks: dict[tuple[str, str], dict[str, Any]] = {}
    cases = review_packet.get("cases")
    if not isinstance(cases, list):
        raise SemanticAdjudicationError("review packet is missing cases")
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        dataset_item_id = str(case.get("dataset_item_id") or "")
        if not dataset_item_id:
            raise SemanticAdjudicationError("review packet case missing dataset_item_id")
        pending = case.get("pending_manual_checks")
        if not isinstance(pending, list):
            raise SemanticAdjudicationError(f"case {dataset_item_id} missing pending_manual_checks")
        for check_name in pending:
            key = (dataset_item_id, str(check_name))
            checks[key] = {
                "dataset_item_id": dataset_item_id,
                "source_trace_id": case.get("source_trace_id"),
                "title": case.get("title"),
                "check_name": str(check_name),
                "candidate_output_preview": case.get("candidate_output_preview", ""),
                "evidence_summary": case.get("evidence_summary", {}),
            }
    return checks


def _iter_fragment_adjudications(fragments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fragment in fragments:
        reviewer = str(fragment.get("reviewer") or "unknown")
        case_adjudications = fragment.get("case_adjudications")
        if not isinstance(case_adjudications, list):
            raise SemanticAdjudicationError(f"reviewer fragment {reviewer} missing case_adjudications")
        for case_adj in case_adjudications:
            if not isinstance(case_adj, Mapping):
                continue
            dataset_item_id = str(case_adj.get("dataset_item_id") or "")
            check_adjudications = case_adj.get("check_adjudications")
            if not isinstance(check_adjudications, list):
                raise SemanticAdjudicationError(f"case adjudication {dataset_item_id} missing check_adjudications")
            for check_adj in check_adjudications:
                if not isinstance(check_adj, Mapping):
                    continue
                check_name = str(check_adj.get("check_name") or "")
                verdict = str(check_adj.get("verdict") or "").lower()
                if verdict not in _VALID_VERDICTS:
                    raise SemanticAdjudicationError(f"invalid verdict for {dataset_item_id}/{check_name}: {verdict}")
                rows.append({
                    "reviewer": reviewer,
                    "dataset_item_id": dataset_item_id,
                    "check_name": check_name,
                    "verdict": verdict,
                    "rationale": str(check_adj.get("rationale") or ""),
                })
    return rows


def build_adjudication(review_packet: Mapping[str, Any], reviewer_fragments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = _packet_checks(review_packet)
    actual_rows = _iter_fragment_adjudications(reviewer_fragments)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in actual_rows:
        key = (row["dataset_item_id"], row["check_name"])
        if key not in expected:
            raise SemanticAdjudicationError(f"unknown manual check in adjudication: {key[0]}/{key[1]}")
        if key in by_key:
            raise SemanticAdjudicationError(f"duplicate manual check adjudication: {key[0]}/{key[1]}")
        by_key[key] = row

    check_rows: list[dict[str, Any]] = []
    pass_count = fail_count = unclear_count = missing_count = 0
    for key in sorted(expected):
        base = expected[key]
        row = by_key.get(key)
        if row is None:
            missing_count += 1
            verdict = "missing"
            rationale = "no adjudication supplied"
            reviewer = None
        else:
            verdict = row["verdict"]
            rationale = row["rationale"]
            reviewer = row["reviewer"]
            if verdict == "pass":
                pass_count += 1
            elif verdict == "fail":
                fail_count += 1
            elif verdict == "unclear":
                unclear_count += 1
        check_rows.append({
            "dataset_item_id": base["dataset_item_id"],
            "source_trace_id": base["source_trace_id"],
            "title": base["title"],
            "check_name": base["check_name"],
            "verdict": verdict,
            "reviewer": reviewer,
            "rationale": rationale,
            "evidence_depth": "minimized_review_packet_plus_candidate_preview",
        })

    all_passing = bool(expected) and pass_count == len(expected) and fail_count == 0 and unclear_count == 0 and missing_count == 0
    score_policy = (
        {
            "test_passed": "eligible_for_future_write_gate",
            "task_success": "eligible_for_future_write_gate",
            "reason": "all manual semantic checks adjudicated pass; separate explicit Langfuse score-write approval still required",
        }
        if all_passing else
        {
            "test_passed": "do_not_write",
            "task_success": "do_not_write",
            "reason": "manual semantic adjudication is not fully passing",
        }
    )
    return {
        "mode": "semantic_adjudication_no_write",
        "write_enabled": False,
        "summary": {
            "case_count": len(review_packet.get("cases", [])) if isinstance(review_packet.get("cases"), list) else 0,
            "manual_check_count": len(expected),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "unclear_count": unclear_count,
            "missing_adjudication_count": missing_count,
        },
        "score_policy_v1": score_policy,
        "check_adjudications": check_rows,
        "explicit_non_claims": [
            "This artifact does not write Langfuse scores.",
            "Pass/unclear/fail verdicts are based on minimized review packets and candidate previews, not raw trace/tool payloads.",
            "Future test_passed/task_success score writes require a separate explicit approval gate.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate no-write Langfuse semantic adjudication fragments.")
    parser.add_argument("--review-packet-json", type=Path, required=True)
    parser.add_argument("--reviewer-fragment-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, help="Optional output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_adjudication(
            _load_json(args.review_packet_json),
            [_load_json(path) for path in args.reviewer_fragment_json],
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(rendered)
        return 0
    except SemanticAdjudicationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
