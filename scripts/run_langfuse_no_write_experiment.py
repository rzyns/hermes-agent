#!/usr/bin/env python3
"""Build a Phase 4 Langfuse no-write experiment evidence bundle.

This runner is intentionally local/artifact-only. It does not call Langfuse and
has no live-write flags. It wraps the existing Phase 3/4 plan, replay/eval,
evidence, and adjudication artifacts into a single sanitized evidence bundle
that reviewers can inspect before any later explicit human-gated write path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)
RAW_PAYLOAD_KEYS = {
    "candidate_output",
    "candidate_outputs",
    "candidate_text",
    "input",
    "output",
    "messages",
    "trace_payload",
    "tool_payload",
    "tool_args",
    "tool_output",
}


class NoWriteRunnerError(ValueError):
    """Raised when a no-write evidence bundle cannot be built safely."""


def redact_text(value: Any) -> str:
    text = str(value)
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: (match.group(1) if match.groups() else "") + "[REDACTED]", text)
    return text


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_load_json(path: Path) -> Any:
    try:
        return _load_json(path)
    except json.JSONDecodeError:
        return None


def _manifest_cases(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), Mapping) else {}
    cases = dataset.get("cases") if isinstance(dataset, Mapping) else None
    if not isinstance(cases, list) or not all(isinstance(item, Mapping) for item in cases):
        raise NoWriteRunnerError("experiment manifest must contain dataset.cases list")
    return cases


def _artifact_specs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), Mapping) else {}
    specs: list[dict[str, Any]] = []
    if isinstance(inputs, Mapping):
        candidate = inputs.get("candidate") if isinstance(inputs.get("candidate"), Mapping) else {}
        evidence = inputs.get("evidence") if isinstance(inputs.get("evidence"), Mapping) else {}
        for label, container, key in (
            ("candidate", candidate, "candidate_artifacts"),
            ("evidence", evidence, "evidence_artifacts"),
        ):
            artifacts = container.get(key) if isinstance(container, Mapping) else []
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if isinstance(artifact, Mapping) and artifact.get("path"):
                        specs.append({"kind": label, **dict(artifact)})
    return specs


def _scan_secret_findings(value: Any, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _contains_sensitive_name(key):
                findings.append({"path": child_path, "pattern": "sensitive_key_name", "preview": "[REDACTED]"})
            findings.extend(_scan_secret_findings(child, child_path))
        return findings
    if isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(_scan_secret_findings(child, f"{path}[{idx}]"))
        return findings
    if value is None or isinstance(value, (bool, int, float)):
        return findings
    text = str(value)
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append({"path": path, "pattern": name, "preview": redact_text(text)[:160]})
    return findings


def _credential_presence_report(env: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    # Do not use the raw environment variable names as JSON keys here: secret
    # scanners intentionally flag key paths containing words like secret/token.
    # The bundle only needs presence/length booleans for reviewer confidence.
    return {
        "public_key": {
            "present": bool(env.get("LANGFUSE_PUBLIC_KEY") or env.get("HERMES_LANGFUSE_PUBLIC_KEY")),
            "length": len(env.get("LANGFUSE_PUBLIC_KEY") or env.get("HERMES_LANGFUSE_PUBLIC_KEY") or ""),
        },
        "write_credential": {
            "present": bool(env.get("LANGFUSE_SECRET_KEY") or env.get("HERMES_LANGFUSE_SECRET_KEY")),
            "length": len(env.get("LANGFUSE_SECRET_KEY") or env.get("HERMES_LANGFUSE_SECRET_KEY") or ""),
        },
        "host": {
            "present": bool(env.get("LANGFUSE_HOST") or env.get("HERMES_LANGFUSE_BASE_URL")),
            "length": len(env.get("LANGFUSE_HOST") or env.get("HERMES_LANGFUSE_BASE_URL") or ""),
        },
    }


def _candidate_statuses(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    statuses: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        sanitized = {
            "dataset_item_id": str(item.get("dataset_item_id") or ""),
            "status": str(item.get("status") or "unknown"),
        }
        if item.get("source_trace_id"):
            sanitized["source_trace_id"] = str(item.get("source_trace_id"))
        statuses.append(sanitized)
    return statuses


def _contains_sensitive_name(key: Any) -> bool:
    lower_key = str(key).lower()
    return any(part in lower_key for part in ("api_key", "secret", "token", "password", "authorization", "private_key"))


def _compact_scalar_summary(payload_summary: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    summary: dict[str, Any] = {}
    omitted_sensitive_named_count = 0
    for key, value in payload_summary.items():
        if not (isinstance(value, (str, int, float, bool)) or value is None):
            continue
        if _contains_sensitive_name(key):
            omitted_sensitive_named_count += 1
            continue
        summary[str(key)] = value
    return summary, omitted_sensitive_named_count


def _artifact_summary(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(spec.get("path")))
    payload = _safe_load_json(path)
    summary: dict[str, Any] = {
        "kind": spec.get("kind"),
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "json_mode": payload.get("mode") if isinstance(payload, Mapping) else None,
        "write_enabled": payload.get("write_enabled") if isinstance(payload, Mapping) else None,
    }
    expected_sha = spec.get("sha256")
    if expected_sha:
        summary["expected_sha256"] = expected_sha
        summary["sha256_matches_expected"] = expected_sha == summary["sha256"]
        if not summary["sha256_matches_expected"]:
            raise NoWriteRunnerError(f"artifact hash mismatch: {path}")
    if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
        # Copy compact counters/status only; never copy payload-bearing fields or
        # sensitive-looking key names from upstream artifact summaries.
        compact_summary, omitted_sensitive_named_count = _compact_scalar_summary(payload["summary"])
        summary["summary"] = compact_summary
        if omitted_sensitive_named_count:
            summary["omitted_sensitive_named_summary_field_count"] = omitted_sensitive_named_count
    return summary


def _assert_no_raw_payload_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in RAW_PAYLOAD_KEYS:
                raise NoWriteRunnerError(f"runner report would persist raw payload field: {child_path}")
            _assert_no_raw_payload_keys(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_no_raw_payload_keys(child, f"{path}[{idx}]")


def build_no_write_bundle(manifest_json: Path, bundle_dir: Path, *, max_batch_size: int = 30, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    manifest = _load_json(manifest_json)
    if not isinstance(manifest, Mapping):
        raise NoWriteRunnerError("experiment manifest must be a JSON object")
    cases = _manifest_cases(manifest)
    if len(cases) > max_batch_size:
        raise NoWriteRunnerError(f"oversized batch: {len(cases)} cases exceeds limit {max_batch_size}")
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), Mapping) else {}
    write_intent = manifest.get("write_intent") if isinstance(manifest.get("write_intent"), Mapping) else {}
    if safety.get("langfuse_write_default") != "blocked" or write_intent.get("mode") != "no_write" or write_intent.get("live_write_flags_allowed") is not False:
        raise NoWriteRunnerError("manifest is not configured for fail-closed no-write execution")

    specs = _artifact_specs(manifest)
    missing = [str(spec.get("path")) for spec in specs if not Path(str(spec.get("path"))).exists()]
    if missing:
        raise NoWriteRunnerError(f"missing artifact(s): {', '.join(missing)}")

    artifact_summaries = [_artifact_summary(spec) for spec in specs]
    omitted_sensitive_named_summary_field_count = sum(
        int(summary.get("omitted_sensitive_named_summary_field_count") or 0)
        for summary in artifact_summaries
    )
    candidate_results: list[dict[str, Any]] = []
    for spec in specs:
        if spec.get("kind") == "candidate":
            candidate_results.extend(_candidate_statuses(_safe_load_json(Path(str(spec.get("path"))))))

    env_report = _credential_presence_report(env or os.environ)
    command_log = {
        "mode": "phase4_no_write_command_log",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "write_enabled": False,
        "steps": [
            {
                "name": "validate_manifest_and_artifacts",
                "status": "completed",
                "case_count": len(cases),
                "artifact_count": len(artifact_summaries),
            },
            {
                "name": "summarize_candidate_statuses_without_candidate_text",
                "status": "completed",
                "candidate_result_count": len(candidate_results),
            },
            {
                "name": "scan_bundle_for_secret_like_values",
                "status": "completed",
            },
        ],
        "credential_environment": env_report,
    }
    artifact_manifest = {
        "mode": "phase4_no_write_artifact_manifest",
        "source_manifest": str(manifest_json),
        "artifacts": artifact_summaries,
    }
    runner_report = {
        "mode": "phase4_no_write_experiment_runner",
        "runner_status": "completed",
        "write_enabled": False,
        "langfuse_writes_attempted": False,
        "manifest_id": manifest.get("manifest_id"),
        "run_spec_digest": (manifest.get("idempotency") or {}).get("run_spec_digest") if isinstance(manifest.get("idempotency"), Mapping) else None,
        "dataset_name": (manifest.get("dataset") or {}).get("dataset_name") if isinstance(manifest.get("dataset"), Mapping) else None,
        "summary": {
            "case_count": len(cases),
            "input_artifact_count": len(artifact_summaries),
            "missing_artifact_count": 0,
            "candidate_result_count": len(candidate_results),
            "omitted_sensitive_named_summary_field_count": omitted_sensitive_named_summary_field_count,
        },
        "candidate_results": candidate_results,
        "runner_status_log": [
            {"step": "validate", "status": "completed"},
            {"step": "bundle", "status": "completed"},
        ],
    }
    _assert_no_raw_payload_keys(runner_report)

    secret_scan_subject = {
        "command_log": command_log,
        "artifact_manifest": artifact_manifest,
        "runner_report": runner_report,
    }
    findings = _scan_secret_findings(secret_scan_subject)
    secret_scan = {
        "mode": "phase4_no_write_bundle_secret_scan",
        "status": "pass" if not findings else "fail",
        "finding_count": len(findings),
        "findings": findings,
    }
    if findings:
        raise NoWriteRunnerError("secret-like value detected in generated no-write bundle metadata")

    _write_json(bundle_dir / "command-log.json", command_log)
    _write_json(bundle_dir / "artifact-manifest.json", artifact_manifest)
    _write_json(bundle_dir / "runner-report.json", runner_report)
    _write_json(bundle_dir / "secret-scan.json", secret_scan)

    report = {
        "mode": "phase4_no_write_experiment_runner",
        "runner_status": "completed",
        "write_enabled": False,
        "langfuse_writes_attempted": False,
        "summary": runner_report["summary"],
        "artifacts": {
            "command_log": str(bundle_dir / "command-log.json"),
            "artifact_manifest": str(bundle_dir / "artifact-manifest.json"),
            "runner_report": str(bundle_dir / "runner-report.json"),
            "secret_scan": str(bundle_dir / "secret-scan.json"),
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Phase 4 no-write Langfuse experiment evidence bundle.")
    parser.add_argument("--manifest-json", type=Path, required=True, help="Phase 4 experiment manifest JSON")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Directory for generated evidence bundle")
    parser.add_argument("--max-batch-size", type=int, default=30, help="Fail closed above this number of dataset cases")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_no_write_bundle(args.manifest_json, args.bundle_dir, max_batch_size=args.max_batch_size)
    except NoWriteRunnerError as exc:
        parser.error(redact_text(exc))
    cli_report = {
        "mode": report["mode"],
        "runner_status": report["runner_status"],
        "write_enabled": report["write_enabled"],
        "langfuse_writes_attempted": report["langfuse_writes_attempted"],
        "summary": report["summary"],
        "artifacts": report["artifacts"],
    }
    sys.stdout.write(json.dumps(cli_report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
