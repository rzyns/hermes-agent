#!/usr/bin/env python3
"""Extract local evaluator artifact evidence from Hermes/Langfuse trace artifacts.

This helper is deliberately read-only/no-write with respect to Langfuse. It turns
raw or wrapped Langfuse trace JSON into the compact Boolean evidence map consumed
by ``run_langfuse_dataset_eval.py --artifact-evidence-json``. It records counts
and booleans only; raw tool inputs/outputs are never copied into the evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class EvidenceExtractionError(ValueError):
    """Raised when evidence extraction inputs are invalid."""


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
        if isinstance(body.get("artifact_evidence"), dict):
            return []
        if body.get("id") or body.get("observations"):
            return [body]
    return []


def _is_tool_observation(obs: Mapping[str, Any]) -> bool:
    return str(obs.get("type") or "") == "TOOL" or str(obs.get("name") or "").startswith("Tool:")


def _metadata_has_args(metadata: Any) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    for key in ("args", "tool_args", "arguments"):
        if key in metadata and metadata.get(key) is not None:
            return True
    return False


def extract_evidence_from_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    trace_id = str(trace.get("id") or "")
    observations = [obs for obs in (trace.get("observations") or []) if isinstance(obs, Mapping)]
    tool_observations = [obs for obs in observations if _is_tool_observation(obs)]

    tool_count = len(tool_observations)
    output_present_count = 0
    null_output_count = 0
    tool_call_id_present_count = 0
    args_present_count = 0

    for obs in tool_observations:
        if obs.get("output") is None:
            null_output_count += 1
        else:
            output_present_count += 1
        metadata = obs.get("metadata") if isinstance(obs.get("metadata"), Mapping) else {}
        if metadata.get("tool_call_id") not in (None, "") or obs.get("tool_call_id") not in (None, ""):
            tool_call_id_present_count += 1
        if obs.get("input") not in (None, "", {}, []) or _metadata_has_args(metadata):
            args_present_count += 1

    all_outputs_present = tool_count > 0 and output_present_count == tool_count
    all_tool_ids_present = tool_count > 0 and tool_call_id_present_count == tool_count
    all_args_present = tool_count > 0 and args_present_count == tool_count
    all_outputs_null = tool_count > 0 and null_output_count == tool_count

    return {
        "tool_outputs_present": all_outputs_present,
        "tool_call_ids_and_args_present": all_tool_ids_present and all_args_present,
        "tool_null_outputs_zero": tool_count > 0 and null_output_count == 0,
        "tool_null_outputs_recorded": null_output_count > 0,
        "all_sampled_tool_outputs_null": all_outputs_null,
        "tool_correlation_gap_classified": tool_count > 0 and (not all_tool_ids_present or null_output_count > 0),
        "profile_success_tool_failure_distinguished": tool_count > 0 and null_output_count > 0,
        "evidence_artifact_exists": bool(trace_id or trace),
        "summary": {
            "trace_id": trace_id,
            "tool_observations": tool_count,
            "tool_output_present_count": output_present_count,
            "tool_null_output_count": null_output_count,
            "tool_call_id_present_count": tool_call_id_present_count,
            "tool_args_present_count": args_present_count,
        },
    }


def build_evidence_report(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    artifact_evidence: dict[str, dict[str, Any]] = {}
    total_tools = 0
    total_null_outputs = 0
    for idx, trace in enumerate(traces):
        trace_id = str(trace.get("id") or f"trace-{idx + 1}")
        evidence = extract_evidence_from_trace(trace)
        artifact_evidence[trace_id] = evidence
        summary = evidence.get("summary") if isinstance(evidence.get("summary"), Mapping) else {}
        total_tools += int(summary.get("tool_observations", 0) or 0)
        total_null_outputs += int(summary.get("tool_null_output_count", 0) or 0)

    return {
        "mode": "artifact_evidence_extraction_no_write",
        "write_enabled": False,
        "summary": {
            "traces_analyzed": len(traces),
            "tool_observations": total_tools,
            "tool_null_outputs": total_null_outputs,
            "evidence_items": len(artifact_evidence),
        },
        "artifact_evidence": artifact_evidence,
        "next_gate": "Feed artifact_evidence into run_langfuse_dataset_eval.py --artifact-evidence-json; keep Langfuse experiment writes separately gated.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract no-write artifact evidence from local Langfuse trace JSON.")
    parser.add_argument("--input-json", type=Path, required=True, help="Local trace JSON export/wrapped CLI response")
    parser.add_argument("--output-json", type=Path, help="Optional output JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _load_json(args.input_json)
        traces = _items_from_payload(payload)
        report = build_evidence_report(traces)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(rendered)
        return 0
    except EvidenceExtractionError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
