#!/usr/bin/env python3
"""Plan a read-only evaluation pass over a Langfuse dataset.

This script intentionally does not run experiments or create Langfuse run items.
It fetches or loads dataset items, summarizes their reviewed eval contracts, and
emits a deterministic/manual-first execution plan that can be reviewed before any
future experiment write is authorized.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib import error, parse, request

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)
FetchDatasetItems = Callable[..., dict[str, Any]]


class EvalPlanError(ValueError):
    """Raised when a dataset eval plan cannot be built safely."""


def redact_text(value: Any) -> str:
    text = str(value)
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


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


def _credential_report(env: dict[str, str]) -> dict[str, Any]:
    return {
        "public_key_present": bool(env.get("LANGFUSE_PUBLIC_KEY")),
        "public_key_length": len(env.get("LANGFUSE_PUBLIC_KEY", "")),
        "secret_key_present": bool(env.get("LANGFUSE_SECRET_KEY")),
        "secret_key_length": len(env.get("LANGFUSE_SECRET_KEY", "")),
        "host_present": bool(env.get("LANGFUSE_HOST")),
    }


def _iter_secret_findings(value: Any, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            findings.extend(_iter_secret_findings(child, child_path))
        return findings
    if isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(_iter_secret_findings(child, f"{path}[{idx}]"))
        return findings
    if value is None or isinstance(value, (bool, int, float)):
        return []

    text = str(value)
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append({"path": path, "pattern": name, "preview": redact_text(text)[:160]})
    return findings


def extract_dataset_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        items = payload["data"]
    elif isinstance(payload, dict) and isinstance(payload.get("body"), dict) and isinstance(payload["body"].get("data"), list):
        items = payload["body"]["data"]
    else:
        raise EvalPlanError("dataset item payload must be a list or object with data/body.data list")
    if not all(isinstance(item, dict) for item in items):
        raise EvalPlanError("dataset items must be objects")
    return items


def load_dataset_items(path: Path) -> list[dict[str, Any]]:
    return extract_dataset_items(json.loads(path.read_text()))


def _expected_output(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("expectedOutput") or item.get("expected_output") or {}
    return value if isinstance(value, dict) else {}


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata") or {}
    return value if isinstance(value, dict) else {}


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(entry) for entry in value]
    if value is None:
        return []
    return [str(value)]


def _deterministic_check(check_name: str) -> dict[str, str]:
    normalized = check_name.lower().strip()
    if any(token in normalized for token in ("privacy", "secret", "redact")):
        return {"name": check_name, "type": "secret_scan", "target": "candidate_output"}
    if any(token in normalized for token in ("format", "json", "schema")):
        return {"name": check_name, "type": "structured_output_check", "target": "candidate_output"}
    if any(token in normalized for token in ("tool", "pytest", "test")):
        return {"name": check_name, "type": "deterministic_artifact_check", "target": "execution_artifacts"}
    return {"name": check_name, "type": "manual_review", "target": "candidate_output"}


def build_plan(items: Sequence[dict[str, Any]], *, dataset_name: str | None) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    secret_findings: list[dict[str, str]] = []
    active_item_count = 0
    items_with_source_trace_id = 0
    items_with_expected_checks = 0
    items_with_human_review = 0

    for idx, item in enumerate(items):
        item_id = str(item.get("id") or f"dataset-item-{idx + 1}")
        source_trace_id = item.get("sourceTraceId") or item.get("source_trace_id")
        status = str(item.get("status") or "UNKNOWN")
        expected = _expected_output(item)
        metadata = _metadata(item)
        must = _as_string_list(expected.get("must"))
        must_not = _as_string_list(expected.get("must_not"))
        checks = _as_string_list(expected.get("checks"))

        if status.upper() == "ACTIVE":
            active_item_count += 1
        if source_trace_id:
            items_with_source_trace_id += 1
        if checks:
            items_with_expected_checks += 1
        if isinstance(metadata.get("human_review"), dict):
            items_with_human_review += 1

        for finding in _iter_secret_findings(item, "item"):
            secret_findings.append({"dataset_item_id": item_id, **finding})

        contracts.append({
            "dataset_item_id": item_id,
            "source_trace_id": str(source_trace_id) if source_trace_id else None,
            "status": status,
            "promotion_reason": metadata.get("promotion_reason"),
            "must": must,
            "must_not": must_not,
            "checks": checks,
            "deterministic_checks": [_deterministic_check(check) for check in checks],
        })

    return {
        "mode": "read_only_eval_plan",
        "dataset_name": dataset_name,
        "summary": {
            "dataset_item_count": len(items),
            "active_item_count": active_item_count,
            "items_with_source_trace_id": items_with_source_trace_id,
            "items_with_expected_checks": items_with_expected_checks,
            "items_with_human_review": items_with_human_review,
            "secret_findings": len(secret_findings),
        },
        "proposed_experiment": {
            "write_enabled": False,
            "requires_explicit_future_flags": ["--write", "--confirm-experiment-write"],
            "scoring_policy": "deterministic_and_manual_first",
        },
        "contracts": contracts,
        "secret_findings": secret_findings,
        "next_gate": "Review this plan, then implement/run the local deterministic evaluator before any Langfuse experiment write.",
    }


def fetch_dataset_items(env: dict[str, str], *, dataset_name: str, limit: int, page: int = 1) -> dict[str, Any]:
    host = env.get("LANGFUSE_HOST", "").rstrip("/")
    public_key = env.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = env.get("LANGFUSE_SECRET_KEY", "")
    if not host or not public_key or not secret_key:
        raise EvalPlanError("LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY are required for --live-read")

    query = parse.urlencode({"datasetName": dataset_name, "limit": limit, "page": page})
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    req = request.Request(
        f"{host}/api/public/dataset-items?{query}",
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=30) as response:  # nosec B310 - operator supplied Langfuse host
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {"data": []}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise EvalPlanError(f"Langfuse HTTP {exc.code}: {redact_text(body)}") from exc
    except Exception as exc:  # pragma: no cover - exercised through callers with injected fetch function
        raise EvalPlanError(f"Langfuse read failed: {redact_text(exc)}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only evaluation plan for a Langfuse dataset.")
    parser.add_argument("--input-json", type=Path, help="Local dataset-items JSON payload")
    parser.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    parser.add_argument("--output-json", type=Path, help="Optional output path for the generated plan")
    parser.add_argument("--live-read", action="store_true", help="Read dataset items from Langfuse; still performs no writes")
    parser.add_argument("--limit", type=int, default=30, help="Dataset items to fetch in --live-read mode")
    parser.add_argument("--page", type=int, default=1, help="Dataset item page to fetch in --live-read mode")
    parser.add_argument("--env-file", type=Path, help="Optional .env file for live Langfuse credentials")
    return parser


def main(argv: Sequence[str] | None = None, *, fetch: FetchDatasetItems = fetch_dataset_items) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        env = prepare_langfuse_env(args.env_file)
        credential_presence = None
        if args.live_read:
            payload = fetch(env, dataset_name=args.dataset_name, limit=args.limit, page=args.page)
            items = extract_dataset_items(payload)
            credential_presence = _credential_report(env)
        else:
            if not args.input_json:
                raise EvalPlanError("--input-json is required unless --live-read is set")
            items = load_dataset_items(args.input_json)

        plan = build_plan(items, dataset_name=args.dataset_name)
        plan["credential_presence"] = credential_presence
        output = json.dumps(plan, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.write_text(output + "\n")
        print(output)
        return 0
    except (EvalPlanError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"success": False, "error": redact_text(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests
    raise SystemExit(main())
