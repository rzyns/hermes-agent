#!/usr/bin/env python3
"""Run a local, no-write evaluation pass from a Langfuse dataset eval plan.

This script intentionally performs no Langfuse writes. It consumes the reviewed
read-only plan produced by ``plan_langfuse_dataset_eval.py`` plus optional local
candidate outputs and artifact evidence, then emits a privacy-safe local result
bundle. Missing outputs are reported as not-run rather than failures; manual
checks remain pending.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error, request

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)

ARTIFACT_CHECK_KEYS: dict[str, str] = {
    "tool_outputs_present": "tool_outputs_present",
    "tool_call_ids_and_args_present": "tool_call_ids_and_args_present",
    "hanna_tool_null_outputs_recorded": "tool_null_outputs_recorded",
    "discord_tool_output_null_recorded": "tool_null_outputs_recorded",
    "all_sampled_tool_outputs_null": "all_sampled_tool_outputs_null",
    "tool_correlation_gap_classified": "tool_correlation_gap_classified",
    "tool_null_outputs_zero": "tool_null_outputs_zero",
    "profile_success_tool_failure_distinguished": "profile_success_tool_failure_distinguished",
    "evidence_artifact_exists": "evidence_artifact_exists",
}


class LocalEvalError(ValueError):
    """Raised when local eval inputs are invalid."""


LangfusePost = Callable[[str, Mapping[str, Any], Mapping[str, str]], dict[str, Any]]


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip():
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def prepare_langfuse_env(env_file: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if env_file:
        for key, value in _load_env_file(env_file).items():
            env.setdefault(key, value)
    env["LANGFUSE_PUBLIC_KEY"] = env.get("LANGFUSE_PUBLIC_KEY") or env.get("HERMES_LANGFUSE_PUBLIC_KEY", "")
    env["LANGFUSE_SECRET_KEY"] = env.get("LANGFUSE_SECRET_KEY") or env.get("HERMES_LANGFUSE_SECRET_KEY", "")
    env["LANGFUSE_HOST"] = env.get("LANGFUSE_HOST") or env.get("HERMES_LANGFUSE_BASE_URL", "")
    return env


def _default_run_name(dataset_name: str | None) -> str:
    safe_dataset = re.sub(r"[^a-zA-Z0-9_.-]+", "-", dataset_name or "dataset").strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"hermes-local-eval-{safe_dataset}-{stamp}"


def _post_langfuse_json(path: str, body: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    host = str(env.get("LANGFUSE_HOST") or "").rstrip("/")
    public_key = str(env.get("LANGFUSE_PUBLIC_KEY") or "")
    secret_key = str(env.get("LANGFUSE_SECRET_KEY") or "")
    if not host or not public_key or not secret_key:
        raise LocalEvalError("LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY are required for --write")
    basic_auth_value = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        f"{host}{path}",
        data=payload,
        headers={"Authorization": f"Basic {basic_auth_value}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:  # nosec B310 - operator supplied Langfuse host
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")[:500]
        raise LocalEvalError(f"Langfuse HTTP {exc.code}: {redact_text(body_text)}") from exc
    except Exception as exc:  # pragma: no cover - exercised by integration/operational use
        raise LocalEvalError(f"Langfuse write failed: {redact_text(exc)}") from exc


def write_langfuse_experiment_run(
    results: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    env_file: Path | None,
    run_name: str,
    run_description: str,
    post_json: LangfusePost = _post_langfuse_json,
) -> dict[str, Any]:
    if results.get("dataset_name") != "hermes/turn-regression/pilot":
        raise LocalEvalError("refusing write: dataset is outside approved hermes/turn-regression/pilot scope")
    summary = results.get("summary") if isinstance(results.get("summary"), Mapping) else {}
    if int(summary.get("contract_count", 0) or 0) > 30:
        raise LocalEvalError("refusing write: approved pilot write is capped at 30 dataset items")
    if int(summary.get("fail_count", 0) or 0) != 0:
        raise LocalEvalError("refusing write: local eval has failures")
    contracts_by_id = {str(contract.get("dataset_item_id")): contract for contract in plan.get("contracts", []) if isinstance(contract, Mapping)}
    env = prepare_langfuse_env(env_file)
    items: list[dict[str, Any]] = []
    failed = 0
    for item in results.get("items", []):
        if not isinstance(item, Mapping):
            continue
        dataset_item_id = str(item.get("dataset_item_id") or "")
        source_trace_id = str(item.get("source_trace_id") or "")
        if not dataset_item_id or not source_trace_id:
            failed += 1
            items.append({"dataset_item_id": dataset_item_id, "source_trace_id": source_trace_id, "write_status": "skipped", "reason": "missing dataset item id or source trace id"})
            continue
        contract = contracts_by_id.get(dataset_item_id, {})
        body = {
            "runName": run_name,
            "runDescription": run_description,
            "datasetItemId": dataset_item_id,
            "traceId": source_trace_id,
        }
        response = post_json("/api/public/dataset-run-items", body, env)
        items.append({
            "dataset_item_id": dataset_item_id,
            "source_trace_id": source_trace_id,
            "promotion_reason": contract.get("promotion_reason"),
            "local_status": item.get("status"),
            "write_status": "created",
            "response_id": response.get("id"),
            "dataset_run_id": response.get("datasetRunId") or response.get("dataset_run_id"),
        })
    return {
        "mode": "langfuse_experiment_write",
        "write_enabled": True,
        "dataset_name": results.get("dataset_name"),
        "run_name": run_name,
        "run_description": run_description,
        "created_run_item_count": len(items) - failed,
        "failed_run_item_count": failed,
        "items": items,
    }


def redact_text(value: Any) -> str:
    text = str(value)
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def _secret_findings(value: Any) -> list[dict[str, str]]:
    text = str(value)
    findings: list[dict[str, str]] = []
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append({"pattern": name, "preview": redact_text(text)[:160]})
    return findings


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_candidate_outputs(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise LocalEvalError("candidate outputs must be a JSON object keyed by dataset item id")
    if isinstance(payload.get("candidate_outputs"), dict):
        payload = payload["candidate_outputs"]
    outputs: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and "output" in value:
            outputs[str(key)] = str(value["output"])
        elif value is None:
            continue
        else:
            outputs[str(key)] = str(value)
    return outputs


def load_artifact_evidence(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise LocalEvalError("artifact evidence must be a JSON object keyed by dataset item id or source trace id")
    if isinstance(payload.get("artifact_evidence"), dict):
        payload = payload["artifact_evidence"]
    evidence: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise LocalEvalError("artifact evidence entries must be JSON objects")
        evidence[str(key)] = dict(value)
    return evidence


def _contracts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    contracts = plan.get("contracts")
    if not isinstance(contracts, list) or not all(isinstance(item, dict) for item in contracts):
        raise LocalEvalError("eval plan must contain a contracts list")
    return contracts


def _check_name(check: Mapping[str, Any]) -> str:
    return str(check.get("name") or "unnamed_check")


def _check_type(check: Mapping[str, Any]) -> str:
    return str(check.get("type") or "manual_review")


def _evaluate_check(
    check: Mapping[str, Any],
    candidate_output: str | None,
    artifact_evidence: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], int, int, bool]:
    name = _check_name(check)
    check_type = _check_type(check)
    if check_type == "manual_review":
        return {"name": name, "type": check_type, "status": "pending", "reason": "manual review required"}, 0, 0, False
    if check_type == "secret_scan":
        if candidate_output is None:
            return {"name": name, "type": check_type, "status": "not_run", "reason": "candidate output missing"}, 0, 0, False
        findings = _secret_findings(candidate_output)
        return {
            "name": name,
            "type": check_type,
            "status": "fail" if findings else "pass",
            "findings": findings,
        }, (1 if findings else 0), len(findings), True
    if check_type == "deterministic_artifact_check":
        evidence_key = ARTIFACT_CHECK_KEYS.get(name)
        if not evidence_key:
            return {"name": name, "type": check_type, "status": "pending", "reason": "deterministic adapter not implemented"}, 0, 0, False
        if not artifact_evidence:
            return {"name": name, "type": check_type, "status": "not_run", "reason": "artifact evidence missing", "evidence_key": evidence_key}, 0, 0, False
        if evidence_key not in artifact_evidence:
            return {"name": name, "type": check_type, "status": "not_run", "reason": "artifact evidence key missing", "evidence_key": evidence_key}, 0, 0, False
        summary = artifact_evidence.get("summary") if isinstance(artifact_evidence.get("summary"), Mapping) else {}
        if evidence_key == "tool_null_outputs_zero" and int(summary.get("tool_observations", 0) or 0) == 0:
            return {
                "name": name,
                "type": check_type,
                "status": "not_applicable",
                "reason": "no tool observations present in artifact evidence",
                "evidence_key": evidence_key,
                "evidence_summary": summary,
            }, 0, 0, False
        passed = bool(artifact_evidence.get(evidence_key))
        result = {"name": name, "type": check_type, "status": "pass" if passed else "fail", "evidence_key": evidence_key}
        if summary:
            result["evidence_summary"] = summary
        if not passed and evidence_key == "tool_call_ids_and_args_present" and summary:
            result["reason"] = "tool call id and args presence coverage is incomplete"
        return result, (0 if passed else 1), 0, True
    if candidate_output is None:
        return {"name": name, "type": check_type, "status": "not_run", "reason": "candidate output missing"}, 0, 0, False
    return {"name": name, "type": check_type, "status": "pending", "reason": "deterministic adapter not implemented"}, 0, 0, False


def build_local_eval_results(
    plan: Mapping[str, Any],
    *,
    candidate_outputs: Mapping[str, str],
    artifact_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contracts = _contracts(plan)
    result_items: list[dict[str, Any]] = []
    evaluated_count = 0
    pass_count = 0
    fail_count = 0
    pending_manual_count = 0
    needs_manual_review_count = 0
    missing_output_count = 0
    secret_findings = 0

    for idx, contract in enumerate(contracts):
        dataset_item_id = str(contract.get("dataset_item_id") or f"dataset-item-{idx + 1}")
        source_trace_id = contract.get("source_trace_id")
        candidate_output = candidate_outputs.get(dataset_item_id)
        item_artifact_evidence = (artifact_evidence or {}).get(dataset_item_id) or (artifact_evidence or {}).get(str(source_trace_id)) or {}
        if candidate_output is None:
            missing_output_count += 1
        if candidate_output is not None or item_artifact_evidence:
            evaluated_count += 1

        check_results: list[dict[str, Any]] = []
        item_failures = 0
        item_ran = False
        item_pending_manual = False
        for check in contract.get("deterministic_checks") or []:
            if not isinstance(check, Mapping):
                continue
            check_result, failures, findings, ran = _evaluate_check(check, candidate_output, item_artifact_evidence)
            item_failures += failures
            secret_findings += findings
            item_ran = item_ran or ran
            if check_result.get("status") == "pending" and check_result.get("type") == "manual_review":
                pending_manual_count += 1
                item_pending_manual = True
            check_results.append(check_result)

        if item_failures:
            item_status = "fail"
            fail_count += 1
        elif item_ran and item_pending_manual:
            item_status = "needs_manual_review"
            needs_manual_review_count += 1
        elif item_ran:
            item_status = "pass"
            pass_count += 1
        else:
            item_status = "not_run"

        result_items.append({
            "dataset_item_id": dataset_item_id,
            "source_trace_id": str(source_trace_id) if source_trace_id else None,
            "status": item_status,
            "checks": check_results,
        })

    return {
        "mode": "local_eval_no_write",
        "dataset_name": plan.get("dataset_name"),
        "write_enabled": False,
        "requires_explicit_future_flags": ["--write", "--confirm-experiment-write"],
        "summary": {
            "contract_count": len(contracts),
            "evaluated_count": evaluated_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pending_manual_count": pending_manual_count,
            "needs_manual_review_count": needs_manual_review_count,
            "missing_output_count": missing_output_count,
            "secret_findings": secret_findings,
        },
        "items": result_items,
        "next_gate": "Review local results, then add concrete replay/output adapters before any Langfuse experiment write.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a no-write local evaluation pass from a Langfuse eval plan.")
    parser.add_argument("--plan-json", type=Path, required=True, help="Read-only eval plan JSON")
    parser.add_argument("--candidate-outputs-json", type=Path, help="Local candidate outputs keyed by dataset item id")
    parser.add_argument("--artifact-evidence-json", type=Path, help="Local artifact/trace evidence keyed by dataset item id or source trace id")
    parser.add_argument("--output-json", type=Path, help="Optional output path for local eval results")
    parser.add_argument("--write", action="store_true", help="Write one scoped Langfuse dataset experiment run; requires --confirm-experiment-write")
    parser.add_argument("--confirm-experiment-write", action="store_true", help="Second explicit guard required with --write")
    parser.add_argument("--run-name", help="Langfuse dataset run name for --write")
    parser.add_argument("--run-description", default="Hermes local evaluator pilot write: replay outputs are review aids; manual checks remain pending.", help="Langfuse dataset run description for --write")
    parser.add_argument("--env-file", type=Path, help="Optional .env file for Langfuse credentials in --write mode")
    return parser


def main(argv: Sequence[str] | None = None, *, writer: Callable[..., dict[str, Any]] = write_langfuse_experiment_run) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.write != args.confirm_experiment_write:
            raise LocalEvalError("Langfuse writes require both --write and --confirm-experiment-write")
        plan = _load_json(args.plan_json)
        if not isinstance(plan, dict):
            raise LocalEvalError("plan JSON must be an object")
        outputs = load_candidate_outputs(args.candidate_outputs_json)
        artifact_evidence = load_artifact_evidence(args.artifact_evidence_json)
        results = build_local_eval_results(plan, candidate_outputs=outputs, artifact_evidence=artifact_evidence)
        if args.write:
            run_name = args.run_name or _default_run_name(str(results.get("dataset_name") or "dataset"))
            results = dict(results)
            results["write_enabled"] = True
            results["langfuse_write"] = writer(
                results,
                plan=plan,
                env_file=args.env_file,
                run_name=run_name,
                run_description=args.run_description,
            )
        rendered = json.dumps(results, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(rendered)
        return 0
    except LocalEvalError as exc:
        parser.error(redact_text(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
