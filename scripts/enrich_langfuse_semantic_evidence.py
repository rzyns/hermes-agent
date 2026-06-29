#!/usr/bin/env python3
"""Build privacy-preserving semantic evidence for Langfuse adjudication gaps.

This is a local/no-write helper. It enriches minimized trace-shape artifacts with
non-payload evidence labels: trace-name mode, content presence/null modes,
environment/profile taxonomy, tool-output/correlation modes, and observation-id
shape. It never copies raw trace names unless they are from a small static
allowlist and never copies raw inputs or outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


STATIC_TRACE_NAMES = {
    "Hermes turn": "static_hermes_turn",
    "Hermes cli turn": "static_hermes_cli_turn",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _items_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    body = payload.get("body", payload)
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict):
        for key in ("data", "traces", "items"):
            value = body.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if body.get("id") or body.get("observations"):
            return [body]
    return []


def _is_tool_observation(obs: Mapping[str, Any]) -> bool:
    return str(obs.get("type") or "") == "TOOL" or str(obs.get("name") or "").startswith("Tool:")


def _value_mode(value: Any) -> str:
    if value is None:
        return "null"
    if value == "":
        return "blank"
    return "present_redacted"


def _trace_name_mode(name: Any) -> tuple[str, str]:
    if name is None or name == "":
        return "blank", "false_static_or_blank"
    text = str(name)
    if text in STATIC_TRACE_NAMES:
        return STATIC_TRACE_NAMES[text], "false_static_or_blank"
    return "other_redacted", "unknown_non_static_name"


def _environment_profile(environment: Any) -> str:
    text = str(environment or "")
    if text.endswith("-hanna") or "hanna" in text:
        return "hanna"
    if text.endswith("-drugi") or "drugi" in text:
        return "drugi"
    if text.endswith("-default") or text == "local-default":
        return "default"
    if text:
        return "other_redacted"
    return "missing"


def _metadata_has_turn_id(metadata: Any) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    for key in ("turn_id", "turnId", "hermes_turn_id"):
        if metadata.get(key) not in (None, ""):
            return True
    return False


def enrich_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    observations = [obs for obs in (trace.get("observations") or []) if isinstance(obs, Mapping)]
    tool_observations = [obs for obs in observations if _is_tool_observation(obs)]
    output_present_count = sum(1 for obs in tool_observations if obs.get("output") is not None)
    null_output_count = sum(1 for obs in tool_observations if obs.get("output") is None)
    missing_tool_call_id_count = 0
    for obs in tool_observations:
        metadata = obs.get("metadata") if isinstance(obs.get("metadata"), Mapping) else {}
        if metadata.get("tool_call_id") in (None, "") and obs.get("tool_call_id") in (None, ""):
            missing_tool_call_id_count += 1
    tool_count = len(tool_observations)
    observation_id_null_count = sum(1 for obs in observations if obs.get("id") in (None, ""))
    trace_name_mode, prompt_leakage_risk = _trace_name_mode(trace.get("name"))
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), Mapping) else {}
    turn_id_present = _metadata_has_turn_id(metadata)
    return {
        "trace_id": str(trace.get("id") or ""),
        "trace_name_mode": trace_name_mode,
        "trace_name_prompt_leakage_risk": prompt_leakage_risk,
        "environment_profile": _environment_profile(trace.get("environment")),
        "trace_input_mode": _value_mode(trace.get("input")),
        "trace_output_mode": _value_mode(trace.get("output")),
        "turn_id_present": turn_id_present,
        "session_id_present": bool(trace.get("sessionId_present") or trace.get("session_id_present") or trace.get("sessionId")),
        "tool_observation_count": tool_count,
        "tool_output_present_count": output_present_count,
        "tool_null_output_count": null_output_count,
        "tool_output_mode": "no_tools" if tool_count == 0 else ("all_null" if null_output_count == tool_count else ("all_present" if output_present_count == tool_count else "mixed")),
        "tool_id_mode": "no_tools" if tool_count == 0 else ("all_tool_call_ids_present" if missing_tool_call_id_count == 0 else "missing_tool_call_ids"),
        "observation_count": len(observations),
        "observation_id_null_count": observation_id_null_count,
        "observation_id_mode": "all_present" if observations and observation_id_null_count == 0 else ("all_null_or_missing" if observations and observation_id_null_count == len(observations) else ("no_observations" if not observations else "mixed")),
    }


def resolve_checks_from_evidence(check_names: Sequence[str], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile = evidence.get("environment_profile")
    tool_mode = evidence.get("tool_output_mode")
    trace_mode = evidence.get("trace_name_mode")
    for check_name in check_names:
        verdict = "unclear"
        rationale = "enriched evidence still does not contain a sufficient privacy-preserving signal for this semantic check"
        if check_name == "prompt_leakage_in_name_false":
            if evidence.get("trace_name_prompt_leakage_risk") == "false_static_or_blank":
                verdict = "pass"
                rationale = f"trace name mode is {trace_mode}; static/blank mode is a privacy-preserving non-leak signal"
        elif check_name in {"content_mode_classified", "capture_content_mode_summary"}:
            verdict = "pass"
            rationale = f"content modes are trace_input={evidence.get('trace_input_mode')}, trace_output={evidence.get('trace_output_mode')}, tool_output={tool_mode}"
        elif check_name == "trace_name_mode_static":
            if trace_mode in {"static_hermes_turn", "static_hermes_cli_turn", "blank"}:
                verdict = "pass"
                rationale = f"trace name mode is {trace_mode}; no raw trace name required"
        elif check_name == "blank_orphan_hanna_classified":
            if profile == "hanna" and tool_mode == "all_null":
                verdict = "pass"
                rationale = "environment profile is hanna and all tool outputs are null, supporting blank/orphan classification"
        elif check_name == "blank_orphan_drugi_classified":
            if profile == "drugi" and tool_mode == "all_null":
                verdict = "pass"
                rationale = "environment profile is drugi and all tool outputs are null, supporting blank/orphan classification"
        elif check_name == "blank_orphan_default_classified":
            if profile == "default" and tool_mode == "all_null":
                verdict = "pass"
                rationale = "environment profile is default and all tool outputs are null, supporting blank/orphan classification"
        elif check_name == "hanna_profile_taxonomy_detected":
            if profile == "hanna":
                verdict = "pass"
                rationale = "environment profile taxonomy resolves to hanna"
        elif check_name == "missing_turn_id_scoring_gap_recorded":
            if not evidence.get("turn_id_present"):
                verdict = "pass"
                rationale = "no turn-id metadata was present in the enriched shape evidence, supporting the scoring-gap record"
        elif check_name == "turn_id_present":
            if evidence.get("turn_id_present"):
                verdict = "pass"
                rationale = "turn-id metadata is present"
            else:
                rationale = "turn-id metadata is not present in the sanitized trace shape"
        elif check_name == "observation_id_is_null_for_scores":
            if evidence.get("observation_id_null_count", 0) > 0:
                verdict = "pass"
                rationale = "at least one observation id is null/missing"
            else:
                verdict = "unclear"
                rationale = "trace observation ids are present; dataset-run score observationId requires score/readback evidence, not trace observation shape"
        elif check_name == "canonical_improved_trace_classified":
            if tool_mode == "all_present" and evidence.get("tool_id_mode") == "all_tool_call_ids_present":
                verdict = "pass"
                rationale = "tool outputs and tool-call ids are fully present, supporting canonical improved plumbing classification"
        rows.append({"check_name": check_name, "verdict": verdict, "rationale": rationale})
    return rows


def build_enrichment_report(trace_shapes: Any, review_packet: Mapping[str, Any]) -> dict[str, Any]:
    traces = _items_from_payload(trace_shapes)
    by_trace_id = {str(trace.get("id") or ""): enrich_trace(trace) for trace in traces if isinstance(trace, Mapping)}
    case_evidence = []
    for case in review_packet.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        trace_id = str(case.get("source_trace_id") or "")
        evidence = by_trace_id.get(trace_id, {"trace_id": trace_id, "missing_trace_shape": True})
        pending = [str(x) for x in case.get("pending_manual_checks", [])]
        case_evidence.append({
            "dataset_item_id": case.get("dataset_item_id"),
            "source_trace_id": trace_id,
            "title": case.get("title"),
            "enriched_evidence": evidence,
            "check_resolutions": resolve_checks_from_evidence(pending, evidence),
        })
    return {
        "mode": "semantic_evidence_enrichment_no_write",
        "write_enabled": False,
        "summary": {
            "trace_count": len(traces),
            "case_count": len(case_evidence),
            "check_resolution_count": sum(len(c["check_resolutions"]) for c in case_evidence),
        },
        "case_evidence": case_evidence,
        "explicit_non_claims": [
            "No Langfuse writes were performed.",
            "Raw trace names outside the static allowlist, raw inputs, and raw outputs are not copied.",
            "Unclear resolutions remain unclear until supported by richer privacy-preserving evidence or explicit human review.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build no-write privacy-preserving semantic evidence from minimized Langfuse trace shapes.")
    parser.add_argument("--trace-shapes-json", type=Path, required=True)
    parser.add_argument("--review-packet-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_enrichment_report(_load_json(args.trace_shapes_json), _load_json(args.review_packet_json))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
