#!/usr/bin/env python3
"""Build a read-only report-only Langfuse dev-loop health summary.

This helper intentionally consumes existing evidence artifacts only. It does not
query Langfuse, create dataset runs, write scores, enable schedulers, or promote
blocking gates. Use it after an explicitly approved exact-scope LF8/LF13 live
smoke has already produced evidence and optional read-only telemetry checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EXPECTED_AGGREGATE_SCORE_NAME = "lf8_report_only_all_items_passed_v1"
ITEM_EVALUATOR_NAME = "lf8_report_only_semantic_match_v1"
LABEL_EVALUATOR_NAME = "lf8_report_only_label"
WRITE_LIKE_FLAGS = {
    "--write",
    "--confirm-experiment-write",
    "--enable-run-level-aggregate-score",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalize_score_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        scores = payload.get("scores") or payload.get("data") or payload.get("items") or []
    elif isinstance(payload, list):
        scores = payload
    else:
        scores = []
    return [score for score in scores if isinstance(score, dict)] if isinstance(scores, list) else []


def value_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def reconcile_dataset_run_aggregate_score(payload: dict[str, Any], expected_name: str = EXPECTED_AGGREGATE_SCORE_NAME) -> dict[str, Any]:
    """Summarize dataset-run aggregate score readback without raw payloads.

    Langfuse can serialize BOOLEAN score values as numeric 1. Treat 1/1.0 and
    true as pass only when the score name and data type also match the expected
    report-only aggregate score.
    """
    scores = normalize_score_items(payload)
    safe_scores = []
    expected_present = False
    for score in scores:
        name = score.get("name")
        data_type = score.get("dataType") or score.get("data_type")
        value = score.get("value")
        expected_boolean_true_match = name == expected_name and data_type == "BOOLEAN" and value in (True, 1, 1.0)
        safe = {
            "name_matches_expected": name == expected_name,
            "dataType_is_boolean": data_type == "BOOLEAN",
            "value_shape": value_shape(value),
            "expected_boolean_true_match": expected_boolean_true_match,
            "source_is_api": score.get("source") == "API",
            "datasetRunId_present": bool(score.get("datasetRunId") or score.get("dataset_run_id") or score.get("datasetRunId_present")),
        }
        safe_scores.append(safe)
        if expected_boolean_true_match:
            expected_present = True
    return {
        "score_count": len(scores),
        "scores": safe_scores,
        "expected_score_name": expected_name,
        "expected_boolean_true_present": expected_present,
        "boolean_true_value_note": "BOOLEAN score values may appear as native true or numeric 1; accepted only with expected score name/type.",
        "raw_payloads_persisted": False,
    }


def summarize_trace_detail(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return an allowlisted trace-shape summary without raw ids/content."""
    if not isinstance(trace, dict):
        return None
    raw_metadata = trace.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    trace_id = str(trace.get("id")) if trace.get("id") else ""
    raw_observation_types = trace.get("observation_types")
    observation_types: dict[str, Any] = raw_observation_types if isinstance(raw_observation_types, dict) else {}
    known_observation_types = {"CHAIN", "GENERATION", "TOOL"}
    return {
        "id_sha256": hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:16] if trace_id else None,
        "name_present": bool(trace.get("name")),
        "sessionId_present": bool(trace.get("sessionId")),
        "tag_count": len(trace.get("tags") or []) if isinstance(trace.get("tags") or [], list) else None,
        "metadata_shape": {
            "source_present": bool(metadata.get("source")),
            "platform_present": bool(metadata.get("platform")),
            "surface_present": bool(metadata.get("surface")),
            "provider_present": bool(metadata.get("provider")),
            "model_present": bool(metadata.get("model")),
            "api_mode_present": bool(metadata.get("api_mode")),
            "turn_id_present": bool(metadata.get("turn_id")),
        },
        "observation_type_counts": {
            "CHAIN": observation_types.get("CHAIN") if isinstance(observation_types.get("CHAIN"), int) else 0,
            "GENERATION": observation_types.get("GENERATION") if isinstance(observation_types.get("GENERATION"), int) else 0,
            "TOOL": observation_types.get("TOOL") if isinstance(observation_types.get("TOOL"), int) else 0,
            "unexpected_type_count": sum(count for name, count in observation_types.items() if name not in known_observation_types and isinstance(count, int)),
        },
        "tool_observations": int_or_none(trace.get("tool_observations")),
        "tool_null_outputs": int_or_none(trace.get("tool_null_outputs")),
    }


def parse_trace_shape_text(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"provided": False}
    recent: list[dict[str, Any]] = []
    trace_detail: dict[str, Any] | None = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("recent: "):
            try:
                recent.append(json.loads(line[len("recent: "):]))
            except json.JSONDecodeError:
                continue
        elif line.startswith("trace_detail: "):
            try:
                trace_detail = json.loads(line[len("trace_detail: "):])
            except json.JSONDecodeError:
                continue
    blank_roots = sum(1 for item in recent if not item.get("name") and not item.get("sessionId") and not item.get("tags"))
    return {
        "provided": True,
        "artifact_path_sha256": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16],
        "artifact_sha256": sha256_file(path),
        "recent_trace_count_reported": len(recent),
        "sampled_recent_blank_root_count": blank_roots,
        "discord_trace_detail": summarize_trace_detail(trace_detail),
    }


def is_iso_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def parse_tool_output_text(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"provided": False}
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"trace=(\S+) name=(.*?) session=(.*?) created=(\S+) "
        r"observations=(\d+) tool_obs=(\d+) tool_null_outputs=(\d+)"
    )
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        created_at = match.group(4)
        rows.append({
            "trace_id_sha256": hashlib.sha256(match.group(1).encode("utf-8")).hexdigest()[:16],
            "name_present": match.group(2) not in ("''", '""', "None", ""),
            "session_present": match.group(3) not in ("''", '""', "None", ""),
            "createdAt_present": bool(created_at),
            "createdAt_parseable": is_iso_timestamp(created_at),
            "observations": int(match.group(5)),
            "tool_observations": int(match.group(6)),
            "tool_null_outputs": int(match.group(7)),
        })
    return {
        "provided": True,
        "artifact_path_sha256": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16],
        "artifact_sha256": sha256_file(path),
        "sampled_trace_count": len(rows),
        "total_tool_observations": sum(row["tool_observations"] for row in rows),
        "total_tool_null_outputs": sum(row["tool_null_outputs"] for row in rows),
        "rows": rows,
    }


def int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def count_named(mapping: Any, key: str) -> int | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    return int(value) if isinstance(value, int) else None


def summarize_live_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    stop_hits = aggregate.get("stop_condition_hits")
    return {
        "total": aggregate.get("total") if isinstance(aggregate.get("total"), int) else None,
        "expected_vs_actual_label_matches": aggregate.get("expected_vs_actual_label_matches") if isinstance(aggregate.get("expected_vs_actual_label_matches"), int) else None,
        "boolean_evaluator_passes": aggregate.get("boolean_evaluator_passes") if isinstance(aggregate.get("boolean_evaluator_passes"), int) else None,
        "stop_condition_hit_count": len(stop_hits) if isinstance(stop_hits, list) else None,
        "secret_like_hits_in_evidence_summary": aggregate.get("secret_like_hits_in_evidence_summary") if isinstance(aggregate.get("secret_like_hits_in_evidence_summary"), int) else None,
    }


def summarize_item_score_readback(item_readback: dict[str, Any]) -> dict[str, Any]:
    raw_names = item_readback.get("score_name_counts")
    names: dict[str, Any] = raw_names if isinstance(raw_names, dict) else {}
    raw_data_types = item_readback.get("data_type_counts")
    data_types: dict[str, Any] = raw_data_types if isinstance(raw_data_types, dict) else {}
    raw_attempts = item_readback.get("run_readback_attempts")
    attempts: list[Any] = raw_attempts if isinstance(raw_attempts, list) else []
    safe_attempts = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        safe_attempts.append({
            "attempt": attempt.get("attempt") if isinstance(attempt.get("attempt"), int) else None,
            "dataset_run_id_present": bool(attempt.get("dataset_run_id_present")),
            "dataset_run_item_count": attempt.get("dataset_run_item_count") if isinstance(attempt.get("dataset_run_item_count"), int) else None,
            "trace_id_count": attempt.get("trace_id_count") if isinstance(attempt.get("trace_id_count"), int) else None,
        })
    known_names = {ITEM_EVALUATOR_NAME, LABEL_EVALUATOR_NAME}
    known_data_types = {"BOOLEAN", "CATEGORICAL"}
    return {
        "trace_id_count": item_readback.get("trace_id_count") if isinstance(item_readback.get("trace_id_count"), int) else None,
        "total_item_level_score_count": item_readback.get("total_item_level_score_count") if isinstance(item_readback.get("total_item_level_score_count"), int) else None,
        "expected_item_evaluator_count": count_named(names, ITEM_EVALUATOR_NAME),
        "expected_label_evaluator_count": count_named(names, LABEL_EVALUATOR_NAME),
        "unexpected_score_name_count": sum(count for name, count in names.items() if name not in known_names and isinstance(count, int)),
        "boolean_score_count": count_named(data_types, "BOOLEAN"),
        "categorical_score_count": count_named(data_types, "CATEGORICAL"),
        "unexpected_data_type_count": sum(count for name, count in data_types.items() if name not in known_data_types and isinstance(count, int)),
        "readback_attempts": safe_attempts,
    }


def summarize_live_smoke(live_smoke: dict[str, Any]) -> dict[str, Any]:
    live_scope = live_smoke.get("live_mutation_scope", {}) or {}
    aggregate = live_smoke.get("aggregate", {}) or {}
    item_readback = live_smoke.get("item_level_score_readback", {}) or {}
    mutations = live_scope.get("mutations_performed")
    mutation_list = mutations if isinstance(mutations, list) else []
    run_name = live_scope.get("run_name")
    dataset_name = live_scope.get("dataset_name")
    dataset_run_id = live_scope.get("dataset_run_id")
    return {
        "run_name_sha256": hashlib.sha256(str(run_name).encode("utf-8")).hexdigest()[:16] if run_name else None,
        "run_name_matches_report_only_namespace": str(run_name).startswith("lf13-live-dev-loop-report-only-smoke-") if run_name else False,
        "dataset_name_matches_exact_lf8_scope": dataset_name == "hermes/live-evaluator/lf8-pilot-smoke-20260512",
        "dataset_run_id_sha256": hashlib.sha256(str(dataset_run_id).encode("utf-8")).hexdigest()[:16] if dataset_run_id else None,
        "dataset_run_id_present": bool(dataset_run_id),
        "live_mutation_shape": {
            "created_dataset_run_recorded": "created_dataset_run" in mutation_list,
            "mutation_count": len(mutation_list),
        },
        "run_level_aggregate_score_enabled": live_scope.get("run_level_aggregate_score_enabled") is True,
        "aggregate_item_results": summarize_live_aggregate(aggregate),
        "item_score_readback": summarize_item_score_readback(item_readback),
    }


def build_lf8_smoke_review_summary(*, live_smoke: dict[str, Any], source_artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize one exact-scope LF8 smoke artifact without requiring aggregate readback.

    This is intentionally weaker than the full dev-loop health summary: it can
    support a PASS_WITH_CAVEATS report-only smoke verdict, but never a blocking
    gate, scheduler, production score, or run-level aggregate claim.
    """
    live_summary = summarize_live_smoke(live_smoke)
    aggregate_item_results = live_summary["aggregate_item_results"]
    item_score_readback = live_summary["item_score_readback"]
    safety = live_smoke.get("safety_boundary", {}) if isinstance(live_smoke.get("safety_boundary"), dict) else {}
    exact_scope_ok = (
        live_summary["dataset_name_matches_exact_lf8_scope"]
        and live_summary["run_name_matches_report_only_namespace"]
        and live_summary["dataset_run_id_present"]
        and live_summary["live_mutation_shape"]["created_dataset_run_recorded"]
        and live_summary["live_mutation_shape"]["mutation_count"] == 1
    )
    result_ok = (
        aggregate_item_results.get("total") == 5
        and aggregate_item_results.get("expected_vs_actual_label_matches") == 5
        and aggregate_item_results.get("boolean_evaluator_passes") == 5
        and aggregate_item_results.get("stop_condition_hit_count") == 0
        and aggregate_item_results.get("secret_like_hits_in_evidence_summary") == 0
    )
    item_scores_ok = (
        item_score_readback.get("trace_id_count") == 5
        and item_score_readback.get("total_item_level_score_count") == 10
        and item_score_readback.get("expected_item_evaluator_count") == 5
        and item_score_readback.get("expected_label_evaluator_count") == 5
        and item_score_readback.get("unexpected_score_name_count") == 0
        and item_score_readback.get("unexpected_data_type_count") == 0
    )
    safety_ok = (
        safety.get("non_blocking_report_only") is True
        and safety.get("production_trace_or_session_score_writes_performed") is False
        and safety.get("broad_trace_backfill_performed") is False
        and safety.get("scheduler_deploy_restart_performed") is False
        and safety.get("blocking_gate_integration_performed") is False
        and safety.get("raw_private_trace_tool_payloads_included") is False
        and safety.get("credential_or_env_values_persisted") is False
        and safety.get("corpus_broadening_performed") is False
    )
    passed = bool(exact_scope_ok and result_ok and item_scores_ok and safety_ok)
    return {
        "schema_version": "lf8_exact_scope_smoke_review_summary_v1",
        "generated_at_utc": utc_now(),
        "verdict": "PASS_WITH_CAVEATS" if passed else "NEEDS_REVIEW",
        "report_only_non_blocking": True,
        "live_write_performed_by_this_helper": False,
        "langfuse_queries_performed_by_this_helper": False,
        "scope": {
            "dataset_name_matches_exact_lf8_scope": live_summary["dataset_name_matches_exact_lf8_scope"],
            "run_name_matches_report_only_namespace": live_summary["run_name_matches_report_only_namespace"],
            "dataset_run_id_present": live_summary["dataset_run_id_present"],
            "created_exactly_one_dataset_run": live_summary["live_mutation_shape"]["created_dataset_run_recorded"] and live_summary["live_mutation_shape"]["mutation_count"] == 1,
        },
        "result_support": aggregate_item_results,
        "score_readback": {
            **item_score_readback,
            "dataset_run_aggregate_score_enabled": live_summary["run_level_aggregate_score_enabled"],
            "dataset_run_aggregate_score_claimed": False,
        },
        "safety_boundary": {
            "non_blocking_report_only": safety.get("non_blocking_report_only") is True,
            "production_trace_or_session_score_writes_performed": safety.get("production_trace_or_session_score_writes_performed") is True,
            "broad_trace_backfill_performed": safety.get("broad_trace_backfill_performed") is True,
            "scheduler_deploy_restart_performed": safety.get("scheduler_deploy_restart_performed") is True,
            "blocking_gate_integration_performed": safety.get("blocking_gate_integration_performed") is True,
            "raw_private_trace_tool_payloads_included": safety.get("raw_private_trace_tool_payloads_included") is True,
            "credential_or_env_values_persisted": safety.get("credential_or_env_values_persisted") is True,
            "corpus_broadening_performed": safety.get("corpus_broadening_performed") is True,
        },
        "source_artifacts": source_artifacts or [],
        "caveats": [
            "Report-only/non-blocking LF8-04 smoke evidence only.",
            "Dataset-run aggregate score is not required or claimed by this summary.",
            "Langfuse item readback can lag; preserved readback attempts describe the observed shape.",
        ],
        "non_claims": [
            "no blocking gate promotion",
            "no scheduler/deploy/restart authorization",
            "no production trace/session scoring authorization",
            "no broad trace backfill authorization",
            "no run-level aggregate score persistence claim",
            "no raw private payload or credential persistence claim beyond artifact shape checks",
        ],
    }


def build_health_summary(
    *,
    live_smoke: dict[str, Any],
    aggregate_score: dict[str, Any],
    trace_shape: dict[str, Any] | None = None,
    tool_output: dict[str, Any] | None = None,
    source_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trace_shape = trace_shape or {"provided": False}
    tool_output = tool_output or {"provided": False}
    live_summary = summarize_live_smoke(live_smoke)
    aggregate_summary = reconcile_dataset_run_aggregate_score(aggregate_score)
    aggregate_item_results = live_summary["aggregate_item_results"]
    item_score_readback = live_summary["item_score_readback"]
    discord_detail = trace_shape.get("discord_trace_detail") or {}
    telemetry = {
        "trace_shape": trace_shape,
        "tool_output_sample": tool_output,
        "interpretation": {
            "ingestion_healthy": bool(discord_detail.get("sessionId_present") and discord_detail.get("tool_null_outputs") == 0) if trace_shape.get("provided") else None,
            "tool_output_health_sample_passed": (tool_output.get("total_tool_null_outputs") == 0 and tool_output.get("sampled_trace_count", 0) > 0) if tool_output.get("provided") else None,
            "root_metadata_caveat": "Newest traces can show blank root metadata while in flight; this summary records the count but does not promote a gate.",
        },
    }
    telemetry_ok = all(
        value is not False
        for value in (
            telemetry["interpretation"]["ingestion_healthy"],
            telemetry["interpretation"]["tool_output_health_sample_passed"],
        )
    )
    expected_scores_ok = (
        item_score_readback.get("total_item_level_score_count") == 10
        and item_score_readback.get("expected_item_evaluator_count") == 5
        and item_score_readback.get("expected_label_evaluator_count") == 5
        and item_score_readback.get("unexpected_score_name_count") == 0
        and item_score_readback.get("unexpected_data_type_count") == 0
    )
    live_smoke_ok = (
        aggregate_item_results.get("total") == 5
        and aggregate_item_results.get("expected_vs_actual_label_matches") == 5
        and aggregate_item_results.get("boolean_evaluator_passes") == 5
        and aggregate_item_results.get("stop_condition_hit_count") == 0
        and aggregate_item_results.get("secret_like_hits_in_evidence_summary") == 0
        and expected_scores_ok
    )
    passed = bool(telemetry_ok and live_smoke_ok and aggregate_summary["expected_boolean_true_present"])
    return {
        "schema_version": "lf13_langfuse_dev_loop_health_summary_v1",
        "generated_at_utc": utc_now(),
        "status": "report_only_non_blocking_snapshot",
        "scope": "Manual read-only health summary over existing telemetry, exact LF8-04 smoke, and dataset-run aggregate score artifacts.",
        "live_write_performed_by_this_helper": False,
        "langfuse_queries_performed_by_this_helper": False,
        "telemetry_health": telemetry,
        "exact_lf8_live_evaluator_smoke": live_summary,
        "dataset_run_aggregate_score": aggregate_summary,
        "overall_readiness": {
            "manual_dev_loop_health_snapshot_passed": passed,
            "blocking_gate_authorized": False,
            "scheduler_or_watchdog_authorized": False,
            "production_score_authorized": False,
            "recommended_next_step": "Keep this as a manual report-only summary unless a separate approval authorizes live writes, scheduling, or gate design.",
        },
        "source_artifacts": source_artifacts or {},
        "non_actions": [
            "no Langfuse writes",
            "no Langfuse API queries",
            "no production trace/session scoring",
            "no broad trace backfill",
            "no scheduler/deploy/restart",
            "no blocking gate promotion",
            "no raw private payload persistence",
            "no credential/env value persistence",
            "no corpus broadening",
        ],
    }


def source_artifact_summary(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "path_sha256": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16],
            "artifact_sha256": sha256_file(path),
        }
        for index, path in enumerate(paths, start=1)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args_for_guard = list(sys.argv[1:] if argv is None else argv)
    write_flags = sorted(flag for flag in args_for_guard if flag in WRITE_LIKE_FLAGS)
    if write_flags:
        raise SystemExit(f"read-only helper rejects write-like flags: {', '.join(write_flags)}")

    parser = argparse.ArgumentParser(description="Summarize existing Langfuse dev-loop health evidence without Langfuse writes or queries")
    parser.add_argument("--live-smoke-json", type=Path, required=True, help="Existing exact-scope LF8/LF13 live smoke evidence JSON")
    parser.add_argument("--aggregate-score-json", type=Path, help="Existing dataset-run aggregate score readback JSON; omit for LF8 smoke-only review summary")
    parser.add_argument("--trace-shape-text", type=Path, help="Optional output from hermes_langfuse_trace_shape_check.py")
    parser.add_argument("--tool-output-text", type=Path, help="Optional output from hermes_langfuse_recent_tool_output_check.py")
    parser.add_argument("--output-json", type=Path, required=True, help="Write report-only health summary JSON")
    ns = parser.parse_args(args_for_guard)

    source_paths = [ns.live_smoke_json]
    if ns.aggregate_score_json:
        source_paths.append(ns.aggregate_score_json)
    if ns.trace_shape_text:
        source_paths.append(ns.trace_shape_text)
    if ns.tool_output_text:
        source_paths.append(ns.tool_output_text)
    source_artifacts = source_artifact_summary(source_paths)
    live_smoke = load_json(ns.live_smoke_json)

    if ns.aggregate_score_json:
        report = build_health_summary(
            live_smoke=live_smoke,
            aggregate_score=load_json(ns.aggregate_score_json),
            trace_shape=parse_trace_shape_text(ns.trace_shape_text),
            tool_output=parse_tool_output_text(ns.tool_output_text),
            source_artifacts=source_artifacts,
        )
    else:
        report = build_lf8_smoke_review_summary(
            live_smoke=live_smoke,
            source_artifacts=source_artifacts,
        )
    write_json(ns.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
