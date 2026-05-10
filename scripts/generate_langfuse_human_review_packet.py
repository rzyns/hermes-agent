#!/usr/bin/env python3
"""Generate human-digestible Langfuse semantic review packets.

This is a local/no-write helper. It converts evaluator items with pending manual
checks into small plain-prose cases suitable for a hybrid human+agent review.
It deliberately summarizes minimized evidence only and never needs raw trace or
raw tool payloads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class ReviewPacketError(RuntimeError):
    """Raised when a review packet cannot be built safely."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ReviewPacketError(f"Invalid JSON in {path}: {exc}") from exc


def _unwrap_candidate_outputs(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("candidate_outputs", payload)
    if not isinstance(raw, Mapping):
        raise ReviewPacketError("candidate outputs must be a mapping or envelope with candidate_outputs")
    return {str(key): str(value) for key, value in raw.items() if isinstance(value, str)}


def _unwrap_artifact_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("artifact_evidence", payload)
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def _contract_index(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    contracts = plan.get("contracts")
    if not isinstance(contracts, list):
        raise ReviewPacketError("plan must contain a contracts list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for contract in contracts:
        if not isinstance(contract, Mapping):
            continue
        dataset_item_id = str(contract.get("dataset_item_id") or "")
        if dataset_item_id:
            indexed[dataset_item_id] = contract
    return indexed


def _eval_items(eval_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = eval_result.get("items")
    if not isinstance(items, list):
        raise ReviewPacketError("eval result must contain an items list")
    return [item for item in items if isinstance(item, Mapping)]


def _pending_manual_checks(item: Mapping[str, Any]) -> list[str]:
    checks = item.get("checks")
    if not isinstance(checks, list):
        return []
    pending: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("type") == "manual_review" and check.get("status") == "pending":
            name = str(check.get("name") or "manual_review")
            pending.append(name)
    return pending


def _automated_statuses(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = item.get("checks")
    if not isinstance(checks, list):
        return []
    out: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("type") == "manual_review":
            continue
        out.append({
            "name": str(check.get("name") or "unknown"),
            "type": str(check.get("type") or "unknown"),
            "status": str(check.get("status") or "unknown"),
            **({"reason": check.get("reason")} if check.get("reason") else {}),
        })
    return out


def _evidence_summary(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return {}
    summary = evidence.get("summary")
    if isinstance(summary, Mapping):
        allowed_keys = [
            "trace_id",
            "tool_observations",
            "tool_null_output_count",
            "tool_outputs_present_count",
            "tool_call_id_present_count",
            "tool_args_present_count",
        ]
        return {key: summary[key] for key in allowed_keys if key in summary}
    allowed = [
        "tool_observations",
        "tool_null_output_count",
        "tool_outputs_present_count",
        "tool_call_id_present_count",
        "tool_args_present_count",
    ]
    return {key: evidence[key] for key in allowed if key in evidence}


def _preview(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + f"… [truncated {len(stripped) - max_chars} chars]"


def build_review_packet(
    plan: Mapping[str, Any],
    *,
    eval_result: Mapping[str, Any],
    candidate_outputs: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any] | None = None,
    max_candidate_chars: int = 1200,
) -> dict[str, Any]:
    """Build a local/no-write packet of human-review cases."""
    contracts = _contract_index(plan)
    outputs = _unwrap_candidate_outputs(candidate_outputs)
    evidence_by_key = _unwrap_artifact_evidence(artifact_evidence or {})
    cases: list[dict[str, Any]] = []
    pending_count = 0

    for item in _eval_items(eval_result):
        pending = _pending_manual_checks(item)
        if not pending:
            continue
        dataset_item_id = str(item.get("dataset_item_id") or "")
        source_trace_id = str(item.get("source_trace_id") or "")
        contract = contracts.get(dataset_item_id, {})
        promotion_reason = str(contract.get("promotion_reason") or "unknown_reason")
        candidate = outputs.get(dataset_item_id, "")
        evidence = evidence_by_key.get(dataset_item_id) or evidence_by_key.get(source_trace_id) or {}
        pending_count += len(pending)
        first_check = pending[0]
        cases.append({
            "title": f"Case {len(cases) + 1} — {promotion_reason} / {dataset_item_id}",
            "dataset_item_id": dataset_item_id,
            "source_trace_id": source_trace_id,
            "promotion_reason": promotion_reason,
            "human_question": (
                "Does the candidate output make a reasonable semantic judgment for "
                f"`{first_check}`, given only the minimized evidence below?"
            ),
            "plain_prose_context": (
                f"This case was selected because `{dataset_item_id}` still has pending manual semantic review. "
                f"The automated evaluator status is `{item.get('status', 'unknown')}`. "
                "Please judge whether the candidate's conclusion is actually reasonable, not merely whether it repeats the contract."
            ),
            "pending_manual_checks": pending,
            "automated_check_statuses": _automated_statuses(item),
            "candidate_output_preview": _preview(candidate, max_candidate_chars),
            "evidence_summary": _evidence_summary(evidence),
            "review_warnings": [
                "Do not infer from raw trace/tool content; this packet intentionally shows minimized evidence only.",
                "Reject or mark unclear if the candidate merely echoes requirements rather than making a supported judgment.",
            ],
            "requested_human_response": ["accept", "reject", "unclear", "notes"],
        })

    return {
        "mode": "human_semantic_review_packet_no_write",
        "write_enabled": False,
        "summary": {
            "dataset_name": str(plan.get("dataset_name") or ""),
            "case_count": len(cases),
            "pending_manual_check_count": pending_count,
        },
        "cases": cases,
    }


def render_markdown(packet: Mapping[str, Any]) -> str:
    summary = packet.get("summary") if isinstance(packet.get("summary"), Mapping) else {}
    lines = [
        "# Langfuse Human Semantic Review Packet",
        "",
        f"Dataset: `{summary.get('dataset_name', '')}`",
        f"Cases: {summary.get('case_count', 0)}",
        f"Pending manual checks represented: {summary.get('pending_manual_check_count', 0)}",
        "",
        "This packet is designed for hybrid human+agent review. It shows only minimized evidence and candidate-output previews.",
        "",
    ]
    for case in packet.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        lines.extend([
            f"## {case.get('title', 'Untitled case')}",
            "",
            f"**Question:** {case.get('human_question', '')}",
            "",
            str(case.get("plain_prose_context", "")),
            "",
            "**Pending manual checks:**",
        ])
        for check in case.get("pending_manual_checks", []):
            lines.append(f"- `{check}`")
        lines.extend(["", "**Automated check statuses:**"])
        for check in case.get("automated_check_statuses", []):
            if isinstance(check, Mapping):
                line = f"- `{check.get('name')}`: `{check.get('status')}`"
                if check.get("reason"):
                    line += f" — {check.get('reason')}"
                lines.append(line)
        lines.extend([
            "",
            "**Minimized evidence summary:**",
            "```json",
            json.dumps(case.get("evidence_summary", {}), indent=2, sort_keys=True),
            "```",
            "",
            "**Candidate output preview:**",
            "",
            str(case.get("candidate_output_preview", "")),
            "",
            "**Review warnings:**",
        ])
        for warning in case.get("review_warnings", []):
            lines.append(f"- {warning}")
        lines.extend([
            "",
            "Please respond with: `accept`, `reject`, or `unclear`, plus short `notes`.",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--candidate-outputs-json", required=True, type=Path)
    parser.add_argument("--artifact-evidence-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    args = parser.parse_args(argv)

    packet = build_review_packet(
        _load_json(args.plan_json),
        eval_result=_load_json(args.eval_json),
        candidate_outputs=_load_json(args.candidate_outputs_json),
        artifact_evidence=_load_json(args.artifact_evidence_json) if args.artifact_evidence_json else {},
        max_candidate_chars=args.max_candidate_chars,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(packet))
    print(json.dumps(packet["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
