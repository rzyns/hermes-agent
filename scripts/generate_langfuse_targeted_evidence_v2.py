#!/usr/bin/env python3
"""Generate final targeted no-write semantic evidence v2 adjudication fragments.

This helper only promotes formerly-unclear checks when there is an explicit,
privacy-preserving shape signal. It does not read or write Langfuse and does not
copy raw candidate text, trace inputs, tool inputs, tool outputs, or credential
values.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence


SECRET_PATTERNS = (
    re.compile(r"sk-lf-[A-Za-z0-9_-]+"),
    re.compile(r"pk-lf-[A-Za-z0-9_-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I),
    re.compile(r"Basic\s+[A-Za-z0-9+/=]{20,}", re.I),
)

TARGETED_CHECKS = {
    "credential_values_not_printed",
    "positive_comparator_recorded",
    "turn_id_present",
    "root_context_absence_recorded",
    "contrasted_with_canonical_drugi_trace",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _secret_scan_passed(case: Mapping[str, Any]) -> bool:
    for status in case.get("automated_check_statuses", []):
        if not isinstance(status, Mapping):
            continue
        if status.get("type") == "secret_scan" and str(status.get("status")) == "pass":
            return True
    return False


def _candidate_preview_has_secret_pattern(case: Mapping[str, Any]) -> bool:
    preview = str(case.get("candidate_output_preview") or "")
    return any(pattern.search(preview) for pattern in SECRET_PATTERNS)


def _case_by_trace(cases: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(case.get("source_trace_id") or ""): case for case in cases}


def _canonical_drugi_exists(cases: Sequence[Mapping[str, Any]], enriched_by_trace: Mapping[str, Mapping[str, Any]]) -> bool:
    for case in cases:
        trace_id = str(case.get("source_trace_id") or "")
        evidence = enriched_by_trace.get(trace_id, {})
        if case.get("promotion_reason") == "canonical_success" and evidence.get("environment_profile") == "drugi":
            if evidence.get("tool_output_mode") == "all_present" and evidence.get("tool_id_mode") == "all_tool_call_ids_present":
                return True
    return False


def _failure_drugi_exists(cases: Sequence[Mapping[str, Any]], enriched_by_trace: Mapping[str, Mapping[str, Any]]) -> bool:
    for case in cases:
        trace_id = str(case.get("source_trace_id") or "")
        evidence = enriched_by_trace.get(trace_id, {})
        if case.get("promotion_reason") == "failure" and evidence.get("environment_profile") == "drugi":
            if evidence.get("tool_output_mode") == "all_null":
                return True
    return False


def resolve_targeted_checks(case: Mapping[str, Any], enriched_by_trace: Mapping[str, Mapping[str, Any]], all_cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pending = [str(check) for check in case.get("pending_manual_checks", [])]
    trace_id = str(case.get("source_trace_id") or "")
    evidence = enriched_by_trace.get(trace_id, {})
    live_shape = evidence.get("live_metadata_shape_v2") if isinstance(evidence.get("live_metadata_shape_v2"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for check in pending:
        if check not in TARGETED_CHECKS:
            continue
        verdict = "unclear"
        evidence_kind = "unresolved_targeted_gap"
        rationale = "targeted evidence v2 did not find a sufficient privacy-preserving signal"
        if check == "credential_values_not_printed":
            if _secret_scan_passed(case) and not _candidate_preview_has_secret_pattern(case):
                verdict = "pass"
                evidence_kind = "secret_scan_status"
                rationale = "case-level secret scan passed and candidate preview has no known secret-token pattern; raw credential values were not persisted"
        elif check == "positive_comparator_recorded":
            if case.get("promotion_reason") == "canonical_success" and _failure_drugi_exists(all_cases, enriched_by_trace):
                verdict = "pass"
                evidence_kind = "comparator_link_shape"
                rationale = "dataset contains a canonical_success drugi trace and at least one contrasting failure/drugi trace, recorded by IDs/classes only"
        elif check == "contrasted_with_canonical_drugi_trace":
            if case.get("promotion_reason") == "failure" and evidence.get("environment_profile") == "drugi" and _canonical_drugi_exists(all_cases, enriched_by_trace):
                verdict = "pass"
                evidence_kind = "comparator_link_shape"
                rationale = "dataset contains a matching canonical_success drugi comparator with complete tool-output/tool-id shape; no raw payloads used"
        elif check == "turn_id_present":
            if live_shape.get("turn_id_present") is True or evidence.get("turn_id_present") is True:
                verdict = "pass"
                evidence_kind = "whitelisted_metadata_shape"
                rationale = "turn-id presence was observed through a whitelisted metadata-shape boolean"
        elif check == "root_context_absence_recorded":
            if live_shape.get("root_context_absence_recorded") is True:
                verdict = "pass"
                evidence_kind = "whitelisted_metadata_shape"
                rationale = "whitelisted live metadata-shape proof records no root-context keys and marks the absence recorded"
        rows.append({"check_name": check, "verdict": verdict, "evidence_kind": evidence_kind, "rationale": rationale})
    return rows


def _load_enriched_by_trace(enriched_report: Mapping[str, Any], live_metadata_shape: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    by_trace: dict[str, dict[str, Any]] = {}
    for case in enriched_report.get("case_evidence", []):
        if not isinstance(case, Mapping):
            continue
        trace_id = str(case.get("source_trace_id") or "")
        evidence = case.get("enriched_evidence") if isinstance(case.get("enriched_evidence"), Mapping) else {}
        by_trace[trace_id] = dict(evidence)
    if live_metadata_shape:
        shapes = live_metadata_shape.get("trace_metadata_shapes", live_metadata_shape)
        if isinstance(shapes, Mapping):
            for trace_id, shape in shapes.items():
                if isinstance(shape, Mapping):
                    by_trace.setdefault(str(trace_id), {})["live_metadata_shape_v2"] = dict(shape)
    return by_trace


def build_fragment(review_packet: Mapping[str, Any], current_adjudication: Mapping[str, Any], enriched_report: Mapping[str, Any], live_metadata_shape: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cases = [case for case in review_packet.get("cases", []) if isinstance(case, Mapping)]
    cases_by_item = {str(case.get("dataset_item_id") or ""): case for case in cases}
    enriched_by_trace = _load_enriched_by_trace(enriched_report, live_metadata_shape)
    targeted_by_case_check: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        for row in resolve_targeted_checks(case, enriched_by_trace, cases):
            targeted_by_case_check[(str(case.get("dataset_item_id") or ""), row["check_name"])] = row

    case_map: OrderedDict[str, dict[str, Any]] = OrderedDict()
    counts: Counter[str] = Counter()
    updates: Counter[str] = Counter()
    for row in current_adjudication.get("check_adjudications", []):
        if not isinstance(row, Mapping):
            continue
        item_id = str(row.get("dataset_item_id") or "")
        case = cases_by_item.get(item_id, {})
        entry = case_map.setdefault(item_id, {
            "case_index": _case_index(row.get("title")),
            "dataset_item_id": item_id,
            "source_trace_id": row.get("source_trace_id"),
            "check_adjudications": [],
        })
        verdict = str(row.get("verdict") or "unclear")
        rationale = str(row.get("rationale") or "")
        targeted = targeted_by_case_check.get((item_id, str(row.get("check_name") or "")))
        if verdict == "unclear" and targeted and targeted["verdict"] == "pass":
            verdict = "pass"
            rationale = "targeted evidence v2: " + targeted["rationale"]
            updates["unclear_to_pass"] += 1
        elif verdict == "unclear" and targeted:
            updates["still_unclear_targeted"] += 1
        counts[verdict] += 1
        entry["check_adjudications"].append({
            "check_name": row.get("check_name"),
            "verdict": verdict,
            "rationale": rationale,
        })
    return {
        "reviewer": "semantic-targeted-evidence-v2",
        "claim_scope": "final targeted semantic adjudication update using secret-scan statuses, comparator shape links, and whitelisted metadata booleans only; no raw payload persistence; no Langfuse writes",
        "case_adjudications": list(case_map.values()),
        "summary": {"counts": dict(counts), "targeted_updates": dict(updates)},
    }


def _case_index(title: Any) -> int | None:
    text = str(title or "")
    match = re.search(r"Case\s+(\d+)", text)
    return int(match.group(1)) if match else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a no-write targeted evidence v2 adjudication fragment.")
    parser.add_argument("--review-packet-json", type=Path, required=True)
    parser.add_argument("--current-adjudication-json", type=Path, required=True)
    parser.add_argument("--enriched-evidence-json", type=Path, required=True)
    parser.add_argument("--live-metadata-shape-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    live_shape = _load_json(args.live_metadata_shape_json) if args.live_metadata_shape_json else None
    fragment = build_fragment(_load_json(args.review_packet_json), _load_json(args.current_adjudication_json), _load_json(args.enriched_evidence_json), live_shape)
    rendered = json.dumps(fragment, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
