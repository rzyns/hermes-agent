#!/usr/bin/env python3
"""Generate privacy-safe local candidate outputs for Langfuse eval dry-runs.

This script is deliberately local/no-write. It turns a reviewed eval plan plus
optional artifact evidence into compact textual candidate outputs that can be fed
into ``run_langfuse_dataset_eval.py --candidate-outputs-json``. It is not a model
replay and does not copy raw trace/tool inputs or outputs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)


class CandidateOutputError(ValueError):
    """Raised when candidate-output generation inputs are invalid."""


def redact_text(value: Any) -> str:
    text = str(value)
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _contracts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    contracts = plan.get("contracts")
    if not isinstance(contracts, list) or not all(isinstance(item, dict) for item in contracts):
        raise CandidateOutputError("eval plan must contain a contracts list")
    return contracts


def load_artifact_evidence(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CandidateOutputError("artifact evidence must be a JSON object")
    if isinstance(payload.get("artifact_evidence"), dict):
        payload = payload["artifact_evidence"]
    evidence: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise CandidateOutputError("artifact evidence entries must be JSON objects")
        evidence[str(key)] = dict(value)
    return evidence


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [redact_text(entry) for entry in value]
    if value is None:
        return []
    return [redact_text(value)]


def _evidence_for_contract(contract: Mapping[str, Any], artifact_evidence: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    dataset_item_id = str(contract.get("dataset_item_id") or "")
    source_trace_id = str(contract.get("source_trace_id") or "")
    return artifact_evidence.get(dataset_item_id) or artifact_evidence.get(source_trace_id) or {}


def _render_candidate_output(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    dataset_item_id = redact_text(contract.get("dataset_item_id") or "")
    source_trace_id = redact_text(contract.get("source_trace_id") or "")
    promotion_reason = redact_text(contract.get("promotion_reason") or "unknown")
    must = _string_list(contract.get("must"))
    must_not = _string_list(contract.get("must_not"))
    checks = _string_list(contract.get("checks"))
    summary = evidence.get("summary") if isinstance(evidence.get("summary"), Mapping) else {}

    lines = [
        "Local candidate output: privacy-safe trace-quality summary.",
        f"dataset_item_id: {dataset_item_id}",
        f"source_trace_id: {source_trace_id}",
        f"promotion_reason: {promotion_reason}",
        "mode: local_candidate_output_no_write",
        "privacy: raw trace input/output payloads were not copied; credential values are not printed.",
    ]

    if summary:
        lines.append("artifact_evidence_summary:")
        for key in sorted(summary):
            if key == "trace_id":
                continue
            value = summary.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"- {redact_text(key)}: {redact_text(value)}")

    boolean_keys = [key for key, value in evidence.items() if isinstance(value, bool)]
    if boolean_keys:
        lines.append("artifact_evidence_flags:")
        for key in sorted(boolean_keys):
            lines.append(f"- {redact_text(key)}: {str(evidence.get(key)).lower()}")

    if checks:
        lines.append("checks_addressed_locally:")
        for check in checks:
            lines.append(f"- {check}")
    if must:
        lines.append("must:")
        for item in must:
            lines.append(f"- {item}")
    if must_not:
        lines.append("must_not:")
        for item in must_not:
            lines.append(f"- {item}")

    return "\n".join(lines)


def build_candidate_outputs(
    plan: Mapping[str, Any],
    *,
    artifact_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int | None = None,
    source_trace_ids: set[str] | None = None,
) -> dict[str, str]:
    contracts = _contracts(plan)
    outputs: dict[str, str] = {}
    for contract in contracts:
        dataset_item_id = str(contract.get("dataset_item_id") or "")
        if not dataset_item_id:
            continue
        source_trace_id = str(contract.get("source_trace_id") or "")
        if source_trace_ids and source_trace_id not in source_trace_ids:
            continue
        evidence = _evidence_for_contract(contract, artifact_evidence or {})
        outputs[dataset_item_id] = _render_candidate_output(contract, evidence)
        if limit is not None and len(outputs) >= limit:
            break
    return outputs


def build_report(plan: Mapping[str, Any], outputs: Mapping[str, str]) -> dict[str, Any]:
    return {
        "mode": "local_candidate_outputs_no_write",
        "dataset_name": plan.get("dataset_name"),
        "write_enabled": False,
        "summary": {
            "contract_count": len(_contracts(plan)),
            "generated_output_count": len(outputs),
            "raw_payloads_copied": False,
        },
        "candidate_outputs": dict(outputs),
        "next_gate": "Feed candidate_outputs into run_langfuse_dataset_eval.py --candidate-outputs-json; keep Langfuse experiment writes separately gated.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate privacy-safe local candidate outputs from a Langfuse eval plan.")
    parser.add_argument("--plan-json", type=Path, required=True, help="Read-only eval plan JSON")
    parser.add_argument("--artifact-evidence-json", type=Path, help="Artifact evidence JSON from extract_langfuse_artifact_evidence.py")
    parser.add_argument("--output-json", type=Path, help="Optional output path for candidate output envelope")
    parser.add_argument("--limit", type=int, help="Maximum number of candidate outputs to generate")
    parser.add_argument("--source-trace-id", action="append", default=[], help="Only generate output for this source trace id; may be repeated")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _load_json(args.plan_json)
        if not isinstance(plan, dict):
            raise CandidateOutputError("plan JSON must be an object")
        evidence = load_artifact_evidence(args.artifact_evidence_json)
        outputs = build_candidate_outputs(
            plan,
            artifact_evidence=evidence,
            limit=args.limit,
            source_trace_ids=set(args.source_trace_id) if args.source_trace_id else None,
        )
        report = build_report(plan, outputs)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(rendered)
        return 0
    except CandidateOutputError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
