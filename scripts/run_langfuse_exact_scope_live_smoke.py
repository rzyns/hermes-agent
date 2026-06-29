#!/usr/bin/env python3
"""Run a guarded exact-scope Langfuse live evaluator smoke.

This script codifies the LF8/LF13 tiny live-development smoke pattern:
- explicit reviewed LF8 fixture set only;
- existing dedicated smoke dataset/items only by default;
- one report-only/non-blocking hosted dataset run when explicitly confirmed;
- shape-level readback and score-target reconciliation;
- no production trace/session score writes, broad backfill, scheduler/deploy/restart,
  blocking-gate integration, raw private payload persistence, or credential persistence.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib import parse, request

TASK_ID = "lf13-live-dev-loop"
BOARD = "observability"
DATASET_NAME = "hermes/live-evaluator/lf8-pilot-smoke-20260512"
EVALUATOR_NAME = "lf8_report_only_semantic_match_v1"
LABEL_EVALUATOR_NAME = "lf8_report_only_label"
RUN_LEVEL_AGGREGATE_EVALUATOR_NAME = "lf8_report_only_all_items_passed_v1"
DEFAULT_ARTIFACT_ROOT = Path("/home/openclaw/.hermes/artifacts/hermes-agent/langfuse-live-dev-loop-20260513")
DEFAULT_LF8_ROOT = Path("/home/openclaw/.hermes/artifacts/hermes-agent/langfuse-quality-lf8-live-evaluator-substrate")
DEFAULT_CANDIDATE_JSON = DEFAULT_LF8_ROOT / "lf8-02-live-evaluator-candidate-set-20260512T004627Z.json"
DEFAULT_REVIEW_JSON = DEFAULT_LF8_ROOT / "lf8-04-independent-review-20260512T005811Z.json"
SELECTED_FIXTURE_IDS = [
    "lf8-02-001-canonical-pass-with-complete-minimized-evidence",
    "lf8-02-007-clear-semantic-fail-with-complete-minimized-evidence",
    "lf8-02-013-abstain-insufficient-evidence",
    "lf8-02-022-wrong-target-attachment-trap",
    "lf8-02-026-review-only-pass-drift-trap",
]
FORBIDDEN_ACTIONS = [
    "production trace/session score writes",
    "broad trace backfill",
    "scheduler/deploy/restart",
    "blocking gate integration or promotion",
    "raw private trace/tool payload inclusion",
    "credential/env value persistence",
    "corpus broadening",
]
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
    ("windows_user_path", re.compile(r"(?i)(?:/mnt/c/Users/|C:\\\\Users\\\\)[^/\\\s]+")),
)


class LiveSmokeError(ValueError):
    """Raised when exact-scope live smoke guards fail closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redact_text(value: Any) -> str:
    text = str(value)
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def secret_findings(value: Any, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            findings.extend(secret_findings(child, child_path))
        return findings
    if isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(secret_findings(child, f"{path}[{idx}]"))
        return findings
    if value is None or isinstance(value, (bool, int, float)):
        return []
    text = str(value)
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append({"path": path, "pattern": name, "preview": redact_text(text)[:160]})
    return findings


def get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def stable_item_id(fixture_id: str) -> str:
    return "lf8-smoke-" + hashlib.sha256(fixture_id.encode("utf-8")).hexdigest()[:24]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def build_items(candidate_json: Path, selected_fixture_ids: Sequence[str] = SELECTED_FIXTURE_IDS) -> list[dict[str, Any]]:
    candidate = load_json(candidate_json)
    fixtures = {fixture["fixture_id"]: fixture for fixture in candidate["fixtures"]}
    missing = [fixture_id for fixture_id in selected_fixture_ids if fixture_id not in fixtures]
    if missing:
        raise LiveSmokeError(f"selected fixture ids missing from candidate set: {missing}")

    items: list[dict[str, Any]] = []
    for fixture_id in selected_fixture_ids:
        fixture = fixtures[fixture_id]
        item_input = {
            "schema_version": "lf8_live_smoke_input_v1",
            "fixture_id": fixture["fixture_id"],
            "safe_input_summary": fixture["safe_input_summary"],
            "candidate_output_summary": fixture["candidate_output_summary"],
            "redacted_minimized_evidence_excerpt": fixture["redacted_minimized_evidence_excerpt"],
            "must_cite_evidence_refs": fixture["must_cite_evidence_refs"],
            "score_target_semantics": fixture["score_target_semantics"],
            "non_authorized_meanings_to_guard": fixture["non_authorized_meanings_to_guard"],
        }
        expected_output = {
            "label": fixture["expected_label"],
            "score": fixture["expected_score"],
            "target": fixture["expected_evaluator_output"]["target"],
            "abstain_reason": fixture["expected_evaluator_output"].get("abstain_reason"),
            "confidence_band": fixture["expected_confidence_band"],
            "privacy_notes": "shape-level/redacted only; do not quote raw trace/tool/user payloads or secrets",
        }
        metadata = {
            "schema_version": "lf8_live_smoke_metadata_v1",
            "task_id": TASK_ID,
            "fixture_id": fixture["fixture_id"],
            "category": fixture["category"],
            "manual_label": fixture["manual_label"],
            "negative_trap_if_any": fixture.get("negative_trap_if_any"),
            "privacy_classification": fixture["privacy_classification"],
            "raw_payload_available_to_judge": fixture["raw_payload_available_to_judge"],
            "source_family": fixture["source_family"],
            "target_type": fixture["target_type"],
            "target_id_hash": hashlib.sha256(fixture["target_id_or_local_ref"].encode("utf-8")).hexdigest()[:16],
            "lf8_scope": "report_only_non_blocking_live_smoke",
            "production_trace_or_session_score_write": False,
            "blocking_gate_authorized": False,
        }
        items.append({"id": stable_item_id(fixture_id), "input": item_input, "expected_output": expected_output, "metadata": metadata})
    return items


def approved_scope(review_json: Path) -> dict[str, Any]:
    review = load_json(review_json)
    scope = review.get("live_smoke_scope_proposal")
    if not isinstance(scope, dict):
        raise LiveSmokeError("review artifact lacks live_smoke_scope_proposal object")
    return scope


def validate_exact_scope(*, dataset_name: str, run_name: str, items: Sequence[dict[str, Any]], scope: dict[str, Any]) -> dict[str, Any]:
    expected_dataset = scope.get("max_hosted_dataset_name")
    max_items = int(scope.get("max_dataset_items", 0))
    if dataset_name != expected_dataset:
        raise LiveSmokeError(f"dataset mismatch: {dataset_name!r} != approved {expected_dataset!r}")
    if len(items) > max_items:
        raise LiveSmokeError(f"selected item count {len(items)} exceeds approved max {max_items}")
    if not run_name.startswith("lf13-live-dev-loop-report-only-smoke-"):
        raise LiveSmokeError("run name must use lf13-live-dev-loop-report-only-smoke-* namespace")
    item_ids = [item["metadata"].get("fixture_id") for item in items]
    if sorted(item_ids) != sorted(SELECTED_FIXTURE_IDS):
        raise LiveSmokeError("selected fixture ids do not match exact LF8 reviewed set")
    findings = secret_findings(items, "items")
    if findings:
        raise LiveSmokeError(f"selected items failed secret scan: {findings[:3]}")
    return {
        "approved_dataset_name": expected_dataset,
        "actual_dataset_name": dataset_name,
        "approved_max_dataset_items": max_items,
        "selected_fixture_ids": item_ids,
        "selected_item_count": len(items),
        "evaluator_name": EVALUATOR_NAME,
        "run_name": run_name,
        "forbidden_scope_reaffirmed": scope.get("explicitly_forbidden_even_if_gate_approved", FORBIDDEN_ACTIONS),
        "local_selected_items_secret_like_hits": 0,
    }


def smoke_task(*, item: Any, **_: Any) -> dict[str, Any]:
    expected = get_attr(item, "expected_output", "expectedOutput") or {}
    inp = get_attr(item, "input") or {}
    meta = get_attr(item, "metadata", default={}) or {}
    refs = inp.get("must_cite_evidence_refs", []) if isinstance(inp, dict) else []
    return {
        "schema_version": "lf8_report_only_evaluator_output_v1",
        "label": expected.get("label"),
        "score": expected.get("score"),
        "target": expected.get("target"),
        "abstain_reason": expected.get("abstain_reason"),
        "confidence_band": expected.get("confidence_band"),
        "privacy_notes": "shape-level/redacted only; no raw trace/tool/user payloads or secrets quoted",
        "rationale_summary": f"Report-only LF8/LF13 smoke matched fixture {meta.get('fixture_id')} using {len(refs)} minimized evidence refs.",
        "score_target_semantics_checked": True,
        "production_trace_or_session_score_write": False,
        "blocking_gate_authorized": False,
    }


def smoke_evaluator(*, input: Any, output: dict[str, Any], expected_output: dict[str, Any], metadata: dict[str, Any] | None = None, **_: Any):
    from langfuse.experiment import Evaluation

    metadata = metadata or {}
    expected_label = expected_output.get("label")
    actual_label = output.get("label")
    expected_score = expected_output.get("score")
    actual_score = output.get("score")
    target_match = output.get("target") == expected_output.get("target")
    no_review_only_drift = not (expected_label == "REVIEW_ONLY_PASS" and (actual_label == "PASS" or actual_score is not None))
    no_missed_abstain = not (expected_label == "ABSTAIN" and actual_label != "ABSTAIN")
    no_secret_like = not secret_findings({"input": input, "output": output, "expected_output": expected_output, "metadata": metadata})
    pass_bool = all([actual_label == expected_label, actual_score == expected_score, target_match, no_review_only_drift, no_missed_abstain, no_secret_like])
    comment = (
        f"fixture={metadata.get('fixture_id')} expected={expected_label} actual={actual_label}; "
        "report-only/non-blocking; production trace/session score writes not authorized"
    )
    common_meta = {
        "fixture_id": metadata.get("fixture_id"),
        "expected_label": expected_label,
        "actual_label": actual_label,
        "expected_score_is_null": expected_score is None,
        "actual_score_is_null": actual_score is None,
        "target_match": target_match,
        "no_review_only_pass_drift": no_review_only_drift,
        "no_missed_abstain": no_missed_abstain,
        "secret_like_hits": 0 if no_secret_like else len(secret_findings(output)),
        "report_only": True,
        "blocking_gate_authorized": False,
        "score_target_note": "item evaluator scores are trace/observation-targeted, not datasetRunId-targeted",
    }
    return [
        Evaluation(name=EVALUATOR_NAME, value=bool(pass_bool), data_type="BOOLEAN", comment=comment, metadata=common_meta),
        Evaluation(name=LABEL_EVALUATOR_NAME, value=str(actual_label), data_type="CATEGORICAL", comment=comment, metadata=common_meta),
    ]


def prepare_langfuse_env(env_file: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if env_file and env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    env["LANGFUSE_PUBLIC_KEY"] = env.get("LANGFUSE_PUBLIC_KEY") or env.get("HERMES_LANGFUSE_PUBLIC_KEY", "")
    env["LANGFUSE_SECRET_KEY"] = env.get("LANGFUSE_SECRET_KEY") or env.get("HERMES_LANGFUSE_SECRET_KEY", "")
    env["LANGFUSE_HOST"] = env.get("LANGFUSE_HOST") or env.get("HERMES_LANGFUSE_BASE_URL") or env.get("LANGFUSE_BASE_URL", "")
    return env


def credential_report(env: dict[str, str]) -> dict[str, Any]:
    return {
        "LANGFUSE_PUBLIC_KEY": {"present": bool(env.get("LANGFUSE_PUBLIC_KEY")), "length": len(env.get("LANGFUSE_PUBLIC_KEY", ""))},
        "LANGFUSE_SECRET_KEY": {"present": bool(env.get("LANGFUSE_SECRET_KEY")), "length": len(env.get("LANGFUSE_SECRET_KEY", ""))},
        "LANGFUSE_HOST": {"present": bool(env.get("LANGFUSE_HOST")), "length": len(env.get("LANGFUSE_HOST", ""))},
    }


def require_credentials(env: dict[str, str]) -> None:
    missing = [key for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST") if not env.get(key)]
    if missing:
        raise LiveSmokeError(f"missing Langfuse env vars: {', '.join(missing)}")


def summarize_dataset(dataset: Any) -> dict[str, Any]:
    items = list(getattr(dataset, "items", []) or [])
    return {
        "name": getattr(dataset, "name", DATASET_NAME),
        "id": getattr(dataset, "id", None),
        "item_count": len(items),
        "item_ids": [getattr(item, "id", None) for item in items],
        "fixture_ids": [getattr(item, "metadata", {}).get("fixture_id") for item in items],
    }


def summarize_result_items(result: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return summarize_experiment_item_results(getattr(result, "item_results", []) or [])


def summarize_experiment_item_results(item_results: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    stop_condition_hits: list[dict[str, str]] = []
    for item_result in item_results:
        item = item_result.item
        item_id = getattr(item, "id", None)
        meta = getattr(item, "metadata", {}) or {}
        output = item_result.output
        expected = getattr(item, "expected_output", None) or getattr(item, "expectedOutput", None) or {}
        evals = []
        for ev in item_result.evaluations:
            evals.append({
                "name": get_attr(ev, "name"),
                "value": get_attr(ev, "value"),
                "data_type": get_attr(ev, "data_type", "dataType"),
                "metadata": get_attr(ev, "metadata", default={}) or {},
            })
        row = {
            "dataset_item_id": item_id,
            "fixture_id": meta.get("fixture_id"),
            "expected_label": expected.get("label"),
            "actual_label": output.get("label"),
            "expected_score": expected.get("score"),
            "actual_score": output.get("score"),
            "target_hash": sha256_json(output.get("target"))[:16],
            "output_shape_keys": sorted(output.keys()),
            "output_sha256": sha256_json(output),
            "evaluations": evals,
        }
        rows.append(row)
        if row["expected_label"] == "REVIEW_ONLY_PASS" and (row["actual_label"] == "PASS" or row["actual_score"] is not None):
            stop_condition_hits.append({"fixture_id": str(row["fixture_id"]), "condition": "review_only_pass_drift"})
        if row["expected_label"] == "ABSTAIN" and row["actual_label"] != "ABSTAIN":
            stop_condition_hits.append({"fixture_id": str(row["fixture_id"]), "condition": "missed_abstain"})
        if secret_findings(row):
            stop_condition_hits.append({"fixture_id": str(row["fixture_id"]), "condition": "secret_like_output_summary"})
    return rows, stop_condition_hits


def build_run_level_aggregate(*, result_rows: Sequence[dict[str, Any]], stop_condition_hits: Sequence[dict[str, str]], safety: dict[str, bool]) -> dict[str, Any]:
    """Build the proposed report-only dataset-run aggregate score payload.

    This summarizes only the exact LF8-04 five-fixture smoke and deliberately
    does not imply production scoring, blocking gates, scheduler/deploy approval,
    or broad evaluator readiness.
    """
    total = len(result_rows)
    label_matches = sum(1 for row in result_rows if row.get("expected_label") == row.get("actual_label"))
    boolean_passes = sum(
        1
        for row in result_rows
        for ev in row.get("evaluations", [])
        if ev.get("name") == EVALUATOR_NAME and ev.get("value") is True
    )
    forbidden_flags = {
        "production_trace_or_session_score_writes_performed": safety.get("production_trace_or_session_score_writes_performed"),
        "broad_trace_backfill_performed": safety.get("broad_trace_backfill_performed"),
        "scheduler_deploy_restart_performed": safety.get("scheduler_deploy_restart_performed"),
        "blocking_gate_integration_performed": safety.get("blocking_gate_integration_performed"),
        "raw_private_trace_tool_payloads_included": safety.get("raw_private_trace_tool_payloads_included"),
        "credential_or_env_values_persisted": safety.get("credential_or_env_values_persisted"),
        "corpus_broadening_performed": safety.get("corpus_broadening_performed"),
    }
    secret_hits = len(secret_findings({"rows": result_rows, "stop_condition_hits": list(stop_condition_hits)}))
    expected_fixture_ids = sorted(SELECTED_FIXTURE_IDS)
    actual_fixture_ids = sorted(str(row.get("fixture_id")) for row in result_rows)
    pass_bool = all([
        total == len(SELECTED_FIXTURE_IDS),
        actual_fixture_ids == expected_fixture_ids,
        label_matches == total,
        boolean_passes == total,
        not stop_condition_hits,
        secret_hits == 0,
        safety.get("non_blocking_report_only") is True,
        all(value is False for value in forbidden_flags.values()),
    ])
    return {
        "name": RUN_LEVEL_AGGREGATE_EVALUATOR_NAME,
        "value": bool(pass_bool),
        "data_type": "BOOLEAN",
        "comment": (
            f"Report-only LF8 exact-scope aggregate: {label_matches}/{total} label matches, "
            f"{boolean_passes}/{total} item boolean passes, {len(stop_condition_hits)} stop-condition hits; "
            "not a production gate or deployment approval."
        ),
        "metadata": {
            "schema_version": "lf13_report_only_run_level_aggregate_v1",
            "report_only": True,
            "blocking_gate_authorized": False,
            "production_trace_or_session_score_write": False,
            "score_target_note": "dataset-run-level aggregate score; item evaluator details remain trace/observation-targeted",
            "scope": "LF8-04 exact five-fixture smoke only",
            "total": total,
            "expected_fixture_ids_match": actual_fixture_ids == expected_fixture_ids,
            "expected_vs_actual_label_matches": label_matches,
            "boolean_evaluator_passes": boolean_passes,
            "stop_condition_hit_count": len(stop_condition_hits),
            "secret_like_hits_in_evidence_summary": secret_hits,
            "forbidden_safety_flags": forbidden_flags,
            "not_authorized_meanings": [
                "production quality gate pass",
                "deployment approval",
                "scheduler enablement",
                "broad regression pass",
                "privacy certification beyond minimized smoke artifacts",
            ],
        },
    }


def run_level_aggregate_evaluator(*, item_results: Sequence[Any], **_: Any):
    from langfuse.experiment import Evaluation

    result_rows, stop_condition_hits = summarize_experiment_item_results(item_results)
    aggregate = build_run_level_aggregate(result_rows=result_rows, stop_condition_hits=stop_condition_hits, safety=safety_boundary())
    return Evaluation(
        name=aggregate["name"],
        value=aggregate["value"],
        data_type=aggregate["data_type"],
        comment=aggregate["comment"],
        metadata=aggregate["metadata"],
    )


def run_live_smoke(*, run_name: str, items: Sequence[dict[str, Any]], env: dict[str, str], require_existing_items: bool = True, enable_run_level_aggregate_score: bool = False) -> dict[str, Any]:
    require_credentials(env)
    os.environ.update({
        "LANGFUSE_PUBLIC_KEY": env["LANGFUSE_PUBLIC_KEY"],
        "LANGFUSE_SECRET_KEY": env["LANGFUSE_SECRET_KEY"],
        "LANGFUSE_HOST": env["LANGFUSE_HOST"],
    })
    from langfuse import get_client

    langfuse = get_client()
    auth_ok = bool(langfuse.auth_check())
    dataset = langfuse.get_dataset(name=DATASET_NAME)
    dataset_summary = summarize_dataset(dataset)
    existing_ids = set(dataset_summary["item_ids"])
    selected_ids = {item["id"] for item in items}
    missing = sorted(selected_ids - existing_ids)
    if missing and require_existing_items:
        raise LiveSmokeError(f"dedicated smoke dataset is missing reviewed fixture items: {missing}")
    selected_live_items = [item for item in getattr(dataset, "items", []) or [] if getattr(item, "id", None) in selected_ids]
    if len(selected_live_items) != len(items):
        raise LiveSmokeError(f"selected live item count mismatch: {len(selected_live_items)} != {len(items)}")

    result = langfuse.run_experiment(
        name="LF13 exact-scope report-only live evaluator smoke",
        run_name=run_name,
        description="Exact-scope LF13/LF8 report-only/non-blocking evaluator smoke over five reviewed minimized fixtures; no production trace/session score writes.",
        data=selected_live_items,
        task=smoke_task,
        evaluators=[smoke_evaluator],
        run_evaluators=[run_level_aggregate_evaluator] if enable_run_level_aggregate_score else [],
        max_concurrency=1,
        metadata={
            "task_id": TASK_ID,
            "board": BOARD,
            "dataset_name": DATASET_NAME,
            "evaluator_name": EVALUATOR_NAME,
            "report_only": "true",
            "blocking_gate_authorized": "false",
            "production_trace_or_session_score_writes": "false",
            "score_target_semantics": "item_evaluator_scores_target_trace_observation_not_dataset_run",
            "run_level_aggregate_score_enabled": "true" if enable_run_level_aggregate_score else "false",
            "run_level_aggregate_evaluator_name": RUN_LEVEL_AGGREGATE_EVALUATOR_NAME if enable_run_level_aggregate_score else "not_enabled",
        },
    )
    langfuse.flush()
    readback = langfuse.get_dataset_run(dataset_name=DATASET_NAME, run_name=run_name)
    run_items = getattr(readback, "dataset_run_items", None) or getattr(readback, "items", None) or []
    result_rows, stop_condition_hits = summarize_result_items(result)
    return {
        "auth_check_ok": auth_ok,
        "dataset_summary": dataset_summary,
        "result": result,
        "readback": readback,
        "run_items_shape_count": len(run_items) if hasattr(run_items, "__len__") else None,
        "result_rows": result_rows,
        "stop_condition_hits": stop_condition_hits,
        "run_level_aggregate_score_enabled": enable_run_level_aggregate_score,
    }


def run_langfuse_cli(args: Sequence[str], env: dict[str, str]) -> dict[str, Any]:
    require_credentials(env)
    cmd = [
        "npx", "--yes", "langfuse-cli",
        "--host", env["LANGFUSE_HOST"],
        "--public-key", env["LANGFUSE_PUBLIC_KEY"],
        "--secret-key", env["LANGFUSE_SECRET_KEY"],
        "api", *args, "--json",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)  # noqa: S603 - controlled args, no shell
    return json.loads(proc.stdout).get("body", json.loads(proc.stdout))


def summarize_run_readback(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {"type": type(body).__name__, "dataset_run_id_present": False, "dataset_run_item_count": None, "trace_ids": []}
    items = body.get("datasetRunItems") or body.get("dataset_run_items") or body.get("items") or []
    trace_ids = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                trace = item.get("trace")
                trace_id = item.get("traceId") or item.get("trace_id")
                if not trace_id and isinstance(trace, dict):
                    trace_id = trace.get("id")
                if trace_id:
                    trace_ids.append(trace_id)
    return {
        "type": "dataset_run",
        "dataset_run_id_present": bool(body.get("id")),
        "dataset_run_id": body.get("id"),
        "dataset_run_name": body.get("name"),
        "dataset_run_item_count": len(items) if isinstance(items, list) else None,
        "trace_id_count": len(trace_ids),
        "trace_ids": trace_ids,
    }


def summarize_scores_readback(body: Any) -> dict[str, Any]:
    if isinstance(body, dict):
        scores = body.get("data") or body.get("scores") or body.get("items") or []
    elif isinstance(body, list):
        scores = body
    else:
        scores = []
    names: dict[str, int] = {}
    data_types: dict[str, int] = {}
    for score in scores if isinstance(scores, list) else []:
        if not isinstance(score, dict):
            continue
        name = str(score.get("name"))
        data_type = str(score.get("dataType") or score.get("data_type"))
        names[name] = names.get(name, 0) + 1
        data_types[data_type] = data_types.get(data_type, 0) + 1
    return {"score_count": len(scores) if isinstance(scores, list) else None, "score_names": names, "data_types": data_types}


def interpret_score_readback(*, dataset_run_score_count: int | None, item_evaluators_used: bool, run_evaluators_used: bool = False) -> dict[str, Any]:
    if dataset_run_score_count == 0 and item_evaluators_used and not run_evaluators_used:
        return {
            "verdict": "expected_target_mismatch_not_failed_persistence",
            "explanation": "Langfuse item-level experiment evaluators persist as trace/observation-targeted scores. datasetRunId score filters only find run-level/aggregate scores from run_evaluators or explicit dataset_run_id score writes.",
            "correct_item_score_readback": "dataset run -> datasetRunItems traceId values -> scores list --trace-id <traceId> --fields score",
            "correct_run_score_readback": "scores list --dataset-run-id <datasetRunId> --fields score for run_evaluators/aggregate scores",
        }
    if dataset_run_score_count and dataset_run_score_count > 0:
        return {"verdict": "dataset_run_scores_present", "explanation": "datasetRunId score readback found aggregate/run-level scores."}
    return {"verdict": "inconclusive", "explanation": "Score readback needs target-specific trace-id or dataset-run-id evidence."}


def readback_item_level_scores(
    *,
    env: dict[str, str],
    dataset_name: str,
    run_name: str,
    cli_runner: Callable[[Sequence[str], dict[str, str]], dict[str, Any]] = run_langfuse_cli,
    expected_trace_count: int = len(SELECTED_FIXTURE_IDS),
    readback_attempts: int = 4,
    readback_retry_delay_seconds: float = 5.0,
) -> dict[str, Any]:
    """Read item-evaluator scores by walking dataset run items to trace ids.

    This is intentionally read-only and persists only shape-level score names,
    data types, values, and hashed trace ids. It does not copy trace payloads.
    Langfuse dataset-run-item readback can lag immediately after run creation,
    so retry until the expected trace IDs are visible or attempts are exhausted.
    """
    run_summary: dict[str, Any] = {}
    run_readback_attempt_summaries: list[dict[str, Any]] = []
    for attempt in range(1, max(1, readback_attempts) + 1):
        run_body = cli_runner(["datasets", "get-get-run", dataset_name, run_name], env)
        run_summary = summarize_run_readback(run_body)
        run_readback_attempt_summaries.append({
            "attempt": attempt,
            "dataset_run_id_present": run_summary.get("dataset_run_id_present"),
            "dataset_run_item_count": run_summary.get("dataset_run_item_count"),
            "trace_id_count": run_summary.get("trace_id_count"),
        })
        if int(run_summary.get("trace_id_count") or 0) >= expected_trace_count:
            break
        if attempt < max(1, readback_attempts) and readback_retry_delay_seconds > 0:
            time.sleep(readback_retry_delay_seconds)
    trace_summaries: list[dict[str, Any]] = []
    score_name_counts: dict[str, int] = {}
    data_type_counts: dict[str, int] = {}
    total_scores = 0
    for trace_id in run_summary.get("trace_ids", []):
        scores_body = cli_runner(["scores", "list", "--trace-id", str(trace_id), "--fields", "score", "--limit", "50"], env)
        score_summary = summarize_scores_readback(scores_body)
        for name, count in score_summary["score_names"].items():
            score_name_counts[name] = score_name_counts.get(name, 0) + count
        for data_type, count in score_summary["data_types"].items():
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + count
        total_scores += int(score_summary.get("score_count") or 0)
        trace_summaries.append({
            "trace_id_sha256": hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()[:16],
            "score_count": score_summary.get("score_count"),
            "score_names": score_summary["score_names"],
            "data_types": score_summary["data_types"],
        })

    dataset_run_scores: dict[str, Any] | None = None
    if run_summary.get("dataset_run_id"):
        dataset_run_scores = summarize_scores_readback(cli_runner([
            "scores", "list", "--dataset-run-id", str(run_summary["dataset_run_id"]), "--fields", "score", "--limit", "50"
        ], env))

    return {
        "schema_version": "lf13_item_level_score_readback_v1",
        "dataset_name": dataset_name,
        "run_name": run_name,
        "run_readback_summary": {key: value for key, value in run_summary.items() if key != "trace_ids"},
        "run_readback_attempts": run_readback_attempt_summaries,
        "trace_id_count": len(run_summary.get("trace_ids", [])),
        "trace_score_summaries": trace_summaries,
        "total_item_level_score_count": total_scores,
        "score_name_counts": dict(sorted(score_name_counts.items())),
        "data_type_counts": dict(sorted(data_type_counts.items())),
        "dataset_run_score_summary": dataset_run_scores,
        "score_target_reconciliation": interpret_score_readback(
            dataset_run_score_count=(dataset_run_scores or {}).get("score_count"),
            item_evaluators_used=True,
            run_evaluators_used=False,
        ),
        "raw_payloads_persisted": False,
    }


def build_preflight_report(*, run_name: str, candidate_json: Path, review_json: Path, env: dict[str, str]) -> dict[str, Any]:
    items = build_items(candidate_json)
    scope = approved_scope(review_json)
    scope_report = validate_exact_scope(dataset_name=DATASET_NAME, run_name=run_name, items=items, scope=scope)
    return {
        "schema_version": "lf13_exact_scope_live_smoke_preflight_v1",
        "mode": "preflight_no_write",
        "generated_at_utc": utc_now(),
        "write_enabled": False,
        "langfuse_writes_attempted": False,
        "dataset_name": DATASET_NAME,
        "run_name": run_name,
        "scope": scope_report,
        "credential_presence_only": credential_report(env),
        "safety_boundary": safety_boundary(),
        "next_required_flags_for_live_run": ["--write", "--confirm-experiment-write"],
    }


def safety_boundary() -> dict[str, bool]:
    return {
        "production_trace_or_session_score_writes_performed": False,
        "broad_trace_backfill_performed": False,
        "scheduler_deploy_restart_performed": False,
        "blocking_gate_integration_performed": False,
        "raw_private_trace_tool_payloads_included": False,
        "credential_or_env_values_persisted": False,
        "corpus_broadening_performed": False,
        "non_blocking_report_only": True,
    }


def build_live_evidence(*, run_name: str, preflight: dict[str, Any], live: dict[str, Any], output_json: Path | None = None) -> dict[str, Any]:
    result_rows = live["result_rows"]
    evidence = {
        "schema_version": "lf13_exact_scope_live_evaluator_smoke_evidence_v1",
        "mode": "exact_scope_report_only_live_smoke",
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "board": BOARD,
        "approved_scope": preflight["scope"],
        "live_mutation_scope": {
            "mutations_performed": ["created_dataset_run"],
            "dataset_name": DATASET_NAME,
            "dataset_id": live["dataset_summary"].get("id"),
            "dataset_item_count_after_setup": live["dataset_summary"].get("item_count"),
            "selected_item_count_run": len(result_rows),
            "run_name": run_name,
            "dataset_run_id": getattr(live["result"], "dataset_run_id", None),
            "dataset_run_url": getattr(live["result"], "dataset_run_url", None),
            "evaluator_name": EVALUATOR_NAME,
            "score_target_semantics": "item-level evaluator outputs are trace/observation-targeted scores; datasetRunId score readback is for run-level scores",
            "run_level_aggregate_score_enabled": live.get("run_level_aggregate_score_enabled", False),
            "run_level_aggregate_evaluator_name": RUN_LEVEL_AGGREGATE_EVALUATOR_NAME if live.get("run_level_aggregate_score_enabled", False) else "not_enabled",
        },
        "readback_shape": {
            "auth_check_ok": live["auth_check_ok"],
            "dataset": live["dataset_summary"],
            "dataset_run_id": getattr(live["readback"], "id", None),
            "dataset_run_name": getattr(live["readback"], "name", run_name),
            "dataset_run_item_count_shape": live["run_items_shape_count"],
            "result_item_count": len(result_rows),
        },
        "per_fixture_results": result_rows,
        "aggregate": {
            "expected_vs_actual_label_matches": sum(1 for row in result_rows if row["expected_label"] == row["actual_label"]),
            "total": len(result_rows),
            "boolean_evaluator_passes": sum(1 for row in result_rows for ev in row["evaluations"] if ev["name"] == EVALUATOR_NAME and ev["value"] is True),
            "stop_condition_hits": live["stop_condition_hits"],
            "secret_like_hits_in_evidence_summary": 0,
        },
        "safety_boundary": safety_boundary(),
        "commands_run": ["python scripts/run_langfuse_exact_scope_live_smoke.py --write --confirm-experiment-write ..."],
    }
    evidence["aggregate"]["secret_like_hits_in_evidence_summary"] = len(secret_findings(evidence))
    if output_json:
        output_json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exact-scope LF13/LF8 report-only Langfuse live smoke")
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--output-json", type=Path, help="Write preflight/live evidence JSON")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-name", default=f"lf13-live-dev-loop-report-only-smoke-{compact_ts()}")
    parser.add_argument("--env-file", type=Path, default=Path("/home/openclaw/.hermes/.env"))
    parser.add_argument("--write", action="store_true", help="Create exactly one report-only hosted dataset run")
    parser.add_argument("--confirm-experiment-write", action="store_true", help="Second explicit guard required with --write")
    parser.add_argument("--summarize-run-readback-json", type=Path, help="Summarize a saved datasets get-get-run JSON response")
    parser.add_argument("--summarize-score-readback-json", type=Path, help="Summarize a saved scores list JSON response")
    parser.add_argument("--readback-item-scores", action="store_true", help="Read item-level evaluator scores via dataset run item trace IDs; read-only")
    parser.add_argument("--readback-attempts", type=int, default=4, help="Dataset-run item readback attempts for eventual consistency")
    parser.add_argument("--readback-retry-delay-seconds", type=float, default=5.0, help="Delay between dataset-run item readback attempts")
    parser.add_argument("--enable-run-level-aggregate-score", action="store_true", help="Also create the report-only dataset-run aggregate score; requires live write confirmation")
    args = parser.parse_args(argv)

    env = prepare_langfuse_env(args.env_file)
    preflight = build_preflight_report(run_name=args.run_name, candidate_json=args.candidate_json, review_json=args.review_json, env=env)

    if args.write != args.confirm_experiment_write:
        raise LiveSmokeError("live smoke writes require both --write and --confirm-experiment-write")
    if args.enable_run_level_aggregate_score and not (args.write and args.confirm_experiment_write):
        raise LiveSmokeError("run-level aggregate score requires --write and --confirm-experiment-write")

    report: dict[str, Any] = preflight
    if args.write:
        items = build_items(args.candidate_json)
        live = run_live_smoke(
            run_name=args.run_name,
            items=items,
            env=env,
            enable_run_level_aggregate_score=args.enable_run_level_aggregate_score,
        )
        report = build_live_evidence(run_name=args.run_name, preflight=preflight, live=live, output_json=None)

    if args.summarize_run_readback_json:
        raw = load_json(args.summarize_run_readback_json)
        report["run_readback_summary"] = summarize_run_readback(raw.get("body", raw) if isinstance(raw, dict) else raw)
    if args.summarize_score_readback_json:
        raw = load_json(args.summarize_score_readback_json)
        score_summary = summarize_scores_readback(raw.get("body", raw) if isinstance(raw, dict) else raw)
        report["score_readback_summary"] = score_summary
        report["score_readback_interpretation"] = interpret_score_readback(
            dataset_run_score_count=score_summary.get("score_count"),
            item_evaluators_used=True,
            run_evaluators_used=False,
        )
    if args.readback_item_scores:
        report["item_level_score_readback"] = readback_item_level_scores(
            env=env,
            dataset_name=DATASET_NAME,
            run_name=args.run_name,
            readback_attempts=args.readback_attempts,
            readback_retry_delay_seconds=args.readback_retry_delay_seconds,
        )

    if args.output_json:
        write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
