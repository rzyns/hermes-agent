#!/usr/bin/env python3
"""Emit a local read-only LF8/LF13 digest from reviewed artifacts.

This helper reads existing artifact files only. It does not query Langfuse, write
Langfuse state, create cron jobs, authorize schedulers, or promote gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

WRITE_LIKE_FLAGS = {
    "--write",
    "--confirm-experiment-write",
    "--enable-run-level-aggregate-score",
    "--create-cron",
    "--schedule",
}
EXPECTED_PACKET_SCHEMA = "lf8_lf13_review_packet_v1"
EXPECTED_DIGEST_SCHEMA = "lf8_lf13_read_only_digest_v1"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object in {path.name}")
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


def nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def pick_keys(payload: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def latest_packet_path(artifact_root: Path) -> Path | None:
    candidates = [
        path for path in artifact_root.glob("lf8-lf13-review-packet-*.json")
        if not path.name.endswith(".stdout.json") and not path.name.endswith(".verification.json")
    ]
    return sorted(candidates)[-1] if candidates else None


def verification_path_for(packet_path: Path) -> Path:
    return packet_path.with_name(packet_path.name.removesuffix(".json") + ".verification.json")


def boundary_failures(packet: dict[str, Any], verification: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    gates = nested_dict(packet, "gate_recommendation")
    safety = nested_dict(packet, "safety_boundary")
    scores = nested_dict(packet, "score_readback")
    scope = nested_dict(packet, "scope")
    results = nested_dict(packet, "result_support")
    if packet.get("schema_version") != EXPECTED_PACKET_SCHEMA:
        reasons.append("source packet schema is not recognized")
    if packet.get("verdict") != "PASS_WITH_CAVEATS":
        reasons.append("source packet verdict is not PASS_WITH_CAVEATS")
    if packet.get("live_write_performed_by_this_helper") is not False:
        reasons.append("source packet does not prove helper writes stayed false")
    if packet.get("langfuse_queries_performed_by_this_helper") is not False:
        reasons.append("source packet does not prove helper queries stayed false")
    if verification.get("all_checks_passed") is not True:
        reasons.append("source verification did not pass")
    secret_findings = verification.get("secret_scan_findings")
    path_findings = verification.get("path_scan_findings")
    if not isinstance(secret_findings, list) or not isinstance(path_findings, list):
        reasons.append("source verification missing explicit clean secret/path scan lists")
    elif secret_findings or path_findings:
        reasons.append("source verification contains secret/path scan findings")
    required_true = {
        "exact LF8 dataset scope": scope.get("dataset_name_matches_exact_lf8_scope"),
        "report-only run namespace": scope.get("run_name_matches_report_only_namespace"),
        "dataset run id present": scope.get("dataset_run_id_present"),
        "created exactly one dataset run": scope.get("created_exactly_one_dataset_run"),
        "non-blocking report-only boundary": safety.get("non_blocking_report_only"),
    }
    for label, value in required_true.items():
        if value is not True:
            reasons.append(f"{label} is not true")
    required_counts = {
        "item total": (results.get("total"), 5),
        "expected/actual label match count": (results.get("expected_vs_actual_label_matches"), 5),
        "boolean evaluator pass count": (results.get("boolean_evaluator_passes"), 5),
        "stop-condition hit count": (results.get("stop_condition_hit_count"), 0),
        "secret-like evidence hit count": (results.get("secret_like_hits_in_evidence_summary"), 0),
        "trace id count": (scores.get("trace_id_count"), 5),
        "item-level score count": (scores.get("total_item_level_score_count"), 10),
        "expected item evaluator count": (scores.get("expected_item_evaluator_count"), 5),
        "expected label evaluator count": (scores.get("expected_label_evaluator_count"), 5),
    }
    for label, (actual, expected) in required_counts.items():
        if actual != expected:
            reasons.append(f"{label} is not {expected}")
    required_false = {
        "blocking gate authorization": gates.get("blocking_gate_authorized"),
        "scheduler authorization": gates.get("scheduler_or_watchdog_authorized"),
        "production score authorization": gates.get("production_score_authorized"),
        "scope expansion authorization": gates.get("scope_expansion_authorized"),
        "fixture expansion authorization": gates.get("fixture_expansion_authorized"),
        "production score writes": safety.get("production_trace_or_session_score_writes_performed"),
        "broad trace backfill": safety.get("broad_trace_backfill_performed"),
        "scheduler deploy restart": safety.get("scheduler_deploy_restart_performed"),
        "blocking gate integration": safety.get("blocking_gate_integration_performed"),
        "raw private payload inclusion": safety.get("raw_private_trace_tool_payloads_included"),
        "credential/env persistence": safety.get("credential_or_env_values_persisted"),
        "corpus broadening": safety.get("corpus_broadening_performed"),
        "dataset-run aggregate score enabled": scores.get("dataset_run_aggregate_score_enabled"),
        "dataset-run aggregate score claimed": scores.get("dataset_run_aggregate_score_claimed"),
    }
    for label, value in required_false.items():
        if value is not False:
            reasons.append(f"{label} is not false")
    return reasons


def state_payload(state_file: Path | None) -> dict[str, Any]:
    if state_file is None or not state_file.exists():
        return {}
    return load_json(state_file)


def update_state(state_file: Path | None, source_sha: str) -> None:
    if state_file is None:
        return
    write_json(state_file, {"last_source_sha256": source_sha, "updated_at_utc": utc_now()})


def build_digest(*, artifact_root: Path, state_file: Path | None = None) -> dict[str, Any]:
    packet_path = latest_packet_path(artifact_root)
    if packet_path is None:
        return {
            "schema_version": EXPECTED_DIGEST_SCHEMA,
            "generated_at_utc": utc_now(),
            "status": "NEEDS_REVIEW",
            "changed_since_last_digest": True,
            "live_write_performed_by_this_helper": False,
            "langfuse_queries_performed_by_this_helper": False,
            "scheduler_authorized": False,
            "blocking_gate_authorized": False,
            "production_score_authorized": False,
            "scope_expansion_authorized": False,
            "fixture_expansion_authorized": False,
            "fail_closed_reasons": ["no LF8/LF13 review packet artifact found"],
        }
    verification_path = verification_path_for(packet_path)
    packet = load_json(packet_path)
    verification = load_json(verification_path) if verification_path.exists() else {}
    source_sha = sha256_file(packet_path)
    previous_state = state_payload(state_file)
    fail_closed_reasons = boundary_failures(packet, verification)
    status = "PASS_WITH_CAVEATS" if not fail_closed_reasons else "NEEDS_REVIEW"
    digest = {
        "schema_version": EXPECTED_DIGEST_SCHEMA,
        "generated_at_utc": utc_now(),
        "status": status,
        "changed_since_last_digest": previous_state.get("last_source_sha256") != source_sha,
        "live_write_performed_by_this_helper": False,
        "langfuse_queries_performed_by_this_helper": False,
        "scheduler_authorized": False,
        "blocking_gate_authorized": False,
        "production_score_authorized": False,
        "scope_expansion_authorized": False,
        "fixture_expansion_authorized": False,
        "source_packet": {
            "artifact_name": packet_path.name,
            "artifact_sha256": source_sha,
        },
        "source_verification": {
            "artifact_name": verification_path.name,
            "artifact_sha256": sha256_file(verification_path) if verification_path.exists() else None,
            "all_checks_passed": verification.get("all_checks_passed"),
        },
        "scope": pick_keys(nested_dict(packet, "scope"), (
            "dataset_name_matches_exact_lf8_scope",
            "run_name_matches_report_only_namespace",
            "dataset_run_id_present",
            "created_exactly_one_dataset_run",
        )),
        "result_support": pick_keys(nested_dict(packet, "result_support"), (
            "total",
            "expected_vs_actual_label_matches",
            "boolean_evaluator_passes",
            "stop_condition_hit_count",
            "secret_like_hits_in_evidence_summary",
        )),
        "score_readback": pick_keys(nested_dict(packet, "score_readback"), (
            "trace_id_count",
            "total_item_level_score_count",
            "expected_item_evaluator_count",
            "expected_label_evaluator_count",
            "dataset_run_aggregate_score_enabled",
            "dataset_run_aggregate_score_claimed",
        )),
        "source_caveat_count": len(packet.get("caveats", [])) if isinstance(packet.get("caveats"), list) else 0,
        "caveats": [],
        "fail_closed_reasons": fail_closed_reasons,
        "next_gate": "human_review_before_scheduler_or_gate_changes",
    }
    return digest


def render_markdown(digest: dict[str, Any]) -> str:
    results = nested_dict(digest, "result_support")
    scores = nested_dict(digest, "score_readback")
    scope = nested_dict(digest, "scope")
    lines = [
        "# LF8/LF13 Read-Only Digest",
        "",
        f"**Status:** {digest['status']}",
        f"**Changed since last digest:** {digest['changed_since_last_digest']}",
        "",
        "## Boundaries",
        "- No Langfuse writes or queries",
        f"- Scheduler authorized: {digest['scheduler_authorized']}",
        f"- Blocking gate authorized: {digest['blocking_gate_authorized']}",
        f"- Production scoring authorized: {digest['production_score_authorized']}",
        f"- Scope expansion authorized: {digest['scope_expansion_authorized']}",
        f"- Fixture expansion authorized: {digest['fixture_expansion_authorized']}",
        "",
        "## Source",
        f"- Packet: {nested_dict(digest, 'source_packet').get('artifact_name')}",
        f"- Verification passed: {nested_dict(digest, 'source_verification').get('all_checks_passed')}",
        "",
        "## Exact-scope support",
        f"- Exact LF8 dataset scope: {scope.get('dataset_name_matches_exact_lf8_scope')}",
        f"- Created exactly one dataset run: {scope.get('created_exactly_one_dataset_run')}",
        f"- Items: {results.get('total')}",
        f"- Label matches: {results.get('expected_vs_actual_label_matches')}",
        f"- Boolean evaluator passes: {results.get('boolean_evaluator_passes')}",
        f"- Stop-condition hits: {results.get('stop_condition_hit_count')}",
        f"- Secret-like evidence hits: {results.get('secret_like_hits_in_evidence_summary')}",
        f"- Trace IDs: {scores.get('trace_id_count')}",
        f"- Item-level score count: {scores.get('total_item_level_score_count')}",
        f"- Dataset-run aggregate score claimed: {scores.get('dataset_run_aggregate_score_claimed')}",
        "",
    ]
    if digest.get("fail_closed_reasons"):
        lines.append("## Fail-closed reasons")
        lines.extend(f"- {reason}" for reason in digest["fail_closed_reasons"])
        lines.append("")
    lines.extend([
        "## Next gate",
        str(digest["next_gate"]),
        "",
    ])
    return "\n".join(lines)


def write_markdown(path: Path, digest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(digest), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    blocked = sorted(set(raw_args) & WRITE_LIKE_FLAGS)
    if blocked:
        raise SystemExit(f"write-like/scheduler flags are not supported by this read-only helper: {blocked}")
    parser = argparse.ArgumentParser(description="Emit a local read-only LF8/LF13 digest from reviewed artifacts")
    parser.add_argument("--artifact-root", type=Path, required=True, help="Directory containing LF8/LF13 review artifacts")
    parser.add_argument("--output-json", type=Path, required=True, help="Write digest JSON")
    parser.add_argument("--output-md", type=Path, required=True, help="Write digest Markdown")
    parser.add_argument("--state-file", type=Path, help="Optional local watermark state file")
    ns = parser.parse_args(raw_args)
    digest = build_digest(artifact_root=ns.artifact_root, state_file=ns.state_file)
    write_json(ns.output_json, digest)
    write_markdown(ns.output_md, digest)
    if digest["status"] == "PASS_WITH_CAVEATS":
        update_state(ns.state_file, digest["source_packet"]["artifact_sha256"])
    if digest["changed_since_last_digest"] or digest["status"] == "NEEDS_REVIEW":
        print(json.dumps(digest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
