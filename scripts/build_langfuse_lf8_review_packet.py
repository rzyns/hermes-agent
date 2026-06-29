#!/usr/bin/env python3
"""Build a read-only LF8/LF13 Langfuse smoke review packet from existing artifacts.

This helper consumes local evidence artifacts only. It does not query Langfuse,
create runs, write scores, enable schedulers, or promote blocking gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

WRITE_LIKE_FLAGS = {
    "--write",
    "--confirm-experiment-write",
    "--enable-run-level-aggregate-score",
}
EXPECTED_SMOKE_SUMMARY_SCHEMA = "lf8_exact_scope_smoke_review_summary_v1"
EXPECTED_HEALTH_SUMMARY_SCHEMA = "lf13_langfuse_dev_loop_health_summary_v1"
SECRET_LIKE_PATTERN = re.compile(r"(?i)(LANGFUSE_SECRET_KEY|sk-lf-[A-Za-z0-9_-]+|pk-lf-[A-Za-z0-9_-]+)")
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_./-])/(?:tmp|home|mnt|var|etc|opt|Users)/[^\s,;:]+")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_summary(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "artifact_name": path.name,
        "artifact_sha256": sha256_file(path),
        "artifact_size_bytes": path.stat().st_size,
    }


def nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def sanitize_text(value: str) -> str:
    value = SECRET_LIKE_PATTERN.sub("[redacted-secret-like-value]", value)
    return ABSOLUTE_PATH_PATTERN.sub("[redacted-absolute-path]", value)


def sanitized_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [sanitize_text(value) for value in values if isinstance(value, str)]


def pick_keys(payload: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def smoke_summary_sections(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    scope = pick_keys(
        nested_dict(summary, "scope"),
        (
            "dataset_name_matches_exact_lf8_scope",
            "run_name_matches_report_only_namespace",
            "dataset_run_id_present",
            "created_exactly_one_dataset_run",
        ),
    )
    results = pick_keys(
        nested_dict(summary, "result_support"),
        (
            "total",
            "expected_vs_actual_label_matches",
            "boolean_evaluator_passes",
            "stop_condition_hit_count",
            "secret_like_hits_in_evidence_summary",
        ),
    )
    scores = pick_keys(
        nested_dict(summary, "score_readback"),
        (
            "trace_id_count",
            "total_item_level_score_count",
            "expected_item_evaluator_count",
            "expected_label_evaluator_count",
            "dataset_run_aggregate_score_enabled",
            "dataset_run_aggregate_score_claimed",
        ),
    )
    safety = pick_keys(
        nested_dict(summary, "safety_boundary"),
        (
            "non_blocking_report_only",
            "production_trace_or_session_score_writes_performed",
            "broad_trace_backfill_performed",
            "scheduler_deploy_restart_performed",
            "blocking_gate_integration_performed",
            "raw_private_trace_tool_payloads_included",
            "credential_or_env_values_persisted",
            "corpus_broadening_performed",
        ),
    )
    return scope, results, scores, safety


def smoke_summary_passes(summary: dict[str, Any]) -> bool:
    scope, results, scores, safety = smoke_summary_sections(summary)
    return all(
        [
            summary.get("schema_version") == EXPECTED_SMOKE_SUMMARY_SCHEMA,
            summary.get("verdict") == "PASS_WITH_CAVEATS",
            summary.get("report_only_non_blocking") is True,
            summary.get("live_write_performed_by_this_helper") is False,
            summary.get("langfuse_queries_performed_by_this_helper") is False,
            scope.get("dataset_name_matches_exact_lf8_scope") is True,
            scope.get("run_name_matches_report_only_namespace") is True,
            scope.get("dataset_run_id_present") is True,
            scope.get("created_exactly_one_dataset_run") is True,
            results.get("total") == 5,
            results.get("expected_vs_actual_label_matches") == 5,
            results.get("boolean_evaluator_passes") == 5,
            results.get("stop_condition_hit_count") == 0,
            results.get("secret_like_hits_in_evidence_summary") == 0,
            scores.get("trace_id_count") == 5,
            scores.get("total_item_level_score_count") == 10,
            scores.get("expected_item_evaluator_count") == 5,
            scores.get("expected_label_evaluator_count") == 5,
            scores.get("dataset_run_aggregate_score_enabled") is False,
            scores.get("dataset_run_aggregate_score_claimed") is False,
            safety.get("non_blocking_report_only") is True,
            safety.get("production_trace_or_session_score_writes_performed") is False,
            safety.get("broad_trace_backfill_performed") is False,
            safety.get("scheduler_deploy_restart_performed") is False,
            safety.get("blocking_gate_integration_performed") is False,
            safety.get("raw_private_trace_tool_payloads_included") is False,
            safety.get("credential_or_env_values_persisted") is False,
            safety.get("corpus_broadening_performed") is False,
        ]
    )


def build_review_packet(
    *,
    smoke_summary: dict[str, Any],
    reviewed_artifacts: list[dict[str, Any]],
    health_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smoke_ok = smoke_summary_passes(smoke_summary)
    scope, result_support, score_readback, safety_boundary = smoke_summary_sections(smoke_summary)
    health_ok = True
    if health_summary is not None:
        overall_readiness = nested_dict(health_summary, "overall_readiness")
        health_ok = (
            health_summary.get("schema_version") == EXPECTED_HEALTH_SUMMARY_SCHEMA
            and health_summary.get("live_write_performed_by_this_helper") is False
            and health_summary.get("langfuse_queries_performed_by_this_helper") is False
            and overall_readiness.get("manual_dev_loop_health_snapshot_passed") is True
            and overall_readiness.get("blocking_gate_authorized") is False
            and overall_readiness.get("scheduler_or_watchdog_authorized") is False
            and overall_readiness.get("production_score_authorized") is False
        )

    verdict = "PASS_WITH_CAVEATS" if smoke_ok and health_ok else "NEEDS_REVIEW"
    caveats = sanitized_string_list(smoke_summary.get("caveats", []))
    non_claims = []
    required_non_claims = [
        "no blocking gate promotion",
        "no scheduler/deploy/restart authorization",
        "no production trace/session scoring authorization",
        "no broad trace backfill authorization",
        "no run-level aggregate score persistence claim",
    ]
    for non_claim in required_non_claims:
        if non_claim not in non_claims:
            non_claims.append(non_claim)

    return {
        "schema_version": "lf8_lf13_review_packet_v1",
        "generated_at_utc": utc_now(),
        "verdict": verdict,
        "live_write_performed_by_this_helper": False,
        "langfuse_queries_performed_by_this_helper": False,
        "reviewed_artifacts": reviewed_artifacts,
        "scope": scope,
        "result_support": result_support,
        "score_readback": score_readback,
        "safety_boundary": safety_boundary,
        "caveats": caveats,
        "non_claims": non_claims,
        "gate_recommendation": {
            "recommended_next_gate": "human_review_before_any_automation_or_scope_expansion",
            "blocking_gate_authorized": False,
            "scheduler_or_watchdog_authorized": False,
            "production_score_authorized": False,
            "scope_expansion_authorized": False,
            "fixture_expansion_authorized": False,
            "recommended_next_step": "Review this packet manually before any automation, fixture expansion, scheduler, or blocking gate design.",
        },
    }


def title_case_non_claim(non_claim: str) -> str:
    if non_claim.startswith("no "):
        return "No " + non_claim[3:]
    return non_claim[:1].upper() + non_claim[1:]


def render_markdown(packet: dict[str, Any]) -> str:
    scope = packet.get("scope", {}) if isinstance(packet.get("scope"), dict) else {}
    results = packet.get("result_support", {}) if isinstance(packet.get("result_support"), dict) else {}
    scores = packet.get("score_readback", {}) if isinstance(packet.get("score_readback"), dict) else {}
    lines = [
        "# Langfuse LF8/LF13 Review Packet",
        "",
        f"**Verdict:** {packet['verdict']}",
        "",
        "## Scope",
        f"- Exact LF8 dataset scope: {scope.get('dataset_name_matches_exact_lf8_scope')}",
        f"- Report-only run namespace: {scope.get('run_name_matches_report_only_namespace')}",
        f"- Dataset run id present: {scope.get('dataset_run_id_present')}",
        f"- Created exactly one dataset run: {scope.get('created_exactly_one_dataset_run')}",
        "",
        "## Result support",
        f"- Items: {results.get('total')}",
        f"- Expected/actual label matches: {results.get('expected_vs_actual_label_matches')}",
        f"- Boolean evaluator passes: {results.get('boolean_evaluator_passes')}",
        f"- Stop-condition hits: {results.get('stop_condition_hit_count')}",
        f"- Secret-like evidence hits: {results.get('secret_like_hits_in_evidence_summary')}",
        "",
        "## Score readback",
        f"- Trace ids: {scores.get('trace_id_count')}",
        f"- Item-level score count: {scores.get('total_item_level_score_count')}",
        f"- Dataset-run aggregate score claimed: {scores.get('dataset_run_aggregate_score_claimed')}",
        "",
        "## Non-claims",
    ]
    for non_claim in packet.get("non_claims", []):
        lines.append(f"- {title_case_non_claim(str(non_claim))}")
    lines.extend([
        "",
        "## Recommended next gate",
        f"- {packet['gate_recommendation']['recommended_next_gate']}",
        "",
    ])
    return "\n".join(lines)


def write_markdown(path: Path, packet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(packet), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    blocked = sorted(set(raw_args) & WRITE_LIKE_FLAGS)
    if blocked:
        raise SystemExit(f"write-like flags are not supported by this read-only helper: {blocked}")

    parser = argparse.ArgumentParser(description="Build a read-only LF8/LF13 review packet from existing local artifacts")
    parser.add_argument("--smoke-summary-json", type=Path, required=True, help="Existing LF8 smoke-only summary JSON")
    parser.add_argument("--health-summary-json", type=Path, help="Optional existing read-only health summary JSON")
    parser.add_argument("--output-json", type=Path, required=True, help="Write review packet JSON")
    parser.add_argument("--output-md", type=Path, required=True, help="Write review packet Markdown")
    ns = parser.parse_args(raw_args)

    smoke_summary = load_json(ns.smoke_summary_json)
    reviewed_artifacts = [artifact_summary(ns.smoke_summary_json, role="smoke_summary")]
    health_summary = None
    if ns.health_summary_json:
        health_summary = load_json(ns.health_summary_json)
        reviewed_artifacts.append(artifact_summary(ns.health_summary_json, role="health_summary"))

    packet = build_review_packet(
        smoke_summary=smoke_summary,
        health_summary=health_summary,
        reviewed_artifacts=reviewed_artifacts,
    )
    write_json(ns.output_json, packet)
    write_markdown(ns.output_md, packet)
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
