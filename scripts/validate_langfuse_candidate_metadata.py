#!/usr/bin/env python3
"""Validate LF4 Langfuse candidate metadata and privacy screens.

This helper is deliberately local/no-write. It validates candidate metadata for
Phase 4 corpus expansion using only JSON files and emits a compact evidence
report. It does not import Langfuse SDKs, read credentials, or write dataset
items/runs/scores.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "hermes.langfuse.phase4.candidate_metadata.v1"
MODE = "candidate_metadata_privacy_screen_no_write"

TAXONOMY: dict[str, set[str]] = {
    "task_type": {
        "repo_readonly",
        "controlled_local_edit",
        "debugging_or_test_failure",
        "browser_or_web_citation",
        "privacy_or_redaction",
        "skill_or_tool_discipline",
        "side_effect_approval",
        "messaging_or_cron_dry_run",
        "known_previous_failure",
    },
    "tool_profile": {
        "no_tool_or_reasoning_only",
        "file_read",
        "terminal_readonly",
        "terminal_write_local",
        "code_edit_local",
        "web_or_browser_readonly",
        "kanban_coordination",
        "scheduler_or_messaging_mock",
        "langfuse_readonly",
        "langfuse_write_gated_excluded_from_corpus",
    },
    "privacy_class": {
        "public_or_synthetic",
        "local_project_nonsecret",
        "private_sanitized",
        "sensitive_requires_redaction",
        "prohibited_raw_or_secret",
    },
    "side_effect_risk": {
        "none",
        "local_artifact_only",
        "local_repo_change",
        "external_readonly",
        "external_write_requires_approval",
        "destructive_or_credential_change_prohibited",
    },
    "score_reliability": {
        "deterministic",
        "artifact_evidence_backed",
        "manual_review_required",
        "llm_judge_optional_nonblocking",
        "insufficient_or_noisy",
    },
    "regression_tier": {
        "pilot_candidate",
        "core_candidate",
        "blocking_regression",
        "nonblocking_watchlist",
        "quarantine",
        "reject",
    },
}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("openai_style_secret", re.compile(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{16,}")),
    ("authorization_bearer", re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)

RAW_PAYLOAD_MARKERS: tuple[str, ...] = (
    "raw_trace_input",
    "raw_trace_output",
    "raw_trace_payload",
    "raw_tool_payload",
    "raw_tool_observation",
    "raw_input",
    "raw_output",
    "verbatim_trace",
    "unredacted_payload",
)

PRIVATE_IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email_address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("ipv4_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("private_identifier_marker", re.compile(r"(?i)\b(user[_-]?id|account[_-]?id|session[_-]?token|phone|address)\b")),
)

REQUIRED_TOP_LEVEL = (
    "candidate_id",
    "provenance",
    "sanitized_summary",
    "taxonomy",
    "risk",
    "expected_behavior",
    "privacy_notes",
)


class CandidateMetadataError(ValueError):
    """Raised when candidate metadata input cannot be processed."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CandidateMetadataError(f"invalid JSON in {path}: {exc}") from exc


def _unwrap_candidates(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        raw = payload.get("candidates", payload.get("candidate_metadata", payload))
    else:
        raw = payload
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise CandidateMetadataError("candidates JSON must be an object, a list of objects, or an envelope with candidates")
    return list(raw)


def _redact(text: str) -> str:
    redacted = text
    for name, pattern in SECRET_PATTERNS:
        redacted = pattern.sub(f"[{name.upper()}_REDACTED]", redacted)
    return redacted


def _preview(value: Any, max_chars: int = 120) -> str:
    text = _redact(str(value)).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"… [truncated {len(text) - max_chars} chars]"


def _walk(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    items = [(path, value)]
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            items.append((f"{path}.{key_text}", key_text))
            items.extend(_walk(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            items.extend(_walk(child, f"{path}[{idx}]"))
    return items


def _finding(kind: str, path: str, detail: str, value: Any | None = None) -> dict[str, Any]:
    finding: dict[str, Any] = {"kind": kind, "path": path, "detail": detail}
    if value is not None:
        finding["preview"] = _preview(value)
    return finding


def _privacy_findings(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, value in _walk(candidate):
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        for marker in RAW_PAYLOAD_MARKERS:
            if marker in lowered:
                findings.append(_finding("raw_payload_marker", path, f"contains marker `{marker}`", value))
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(value):
                findings.append(_finding("secret_like_value", path, f"matches {name}", value))
        for name, pattern in PRIVATE_IDENTIFIER_PATTERNS:
            if pattern.search(value):
                findings.append(_finding("private_identifier_marker", path, f"matches {name}", value))
    return findings


def _as_mapping(candidate: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = candidate.get(key)
    return value if isinstance(value, Mapping) else {}


def _has_proof_artifact(review_evidence: Mapping[str, Any]) -> bool:
    proof_artifacts = review_evidence.get("proof_artifacts")
    if not isinstance(proof_artifacts, list):
        return False
    for artifact in proof_artifacts:
        if isinstance(artifact, Mapping) and artifact.get("path") and artifact.get("sha256"):
            return True
    return False


def validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "") or "<missing>"
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in candidate:
            errors.append(_finding("missing_required_field", f"$.{key}", f"missing required field `{key}`"))

    taxonomy = _as_mapping(candidate, "taxonomy")
    for axis, allowed in TAXONOMY.items():
        value = taxonomy.get(axis)
        if value is None:
            errors.append(_finding("missing_taxonomy_axis", f"$.taxonomy.{axis}", f"missing taxonomy axis `{axis}`"))
        elif value not in allowed:
            errors.append(_finding("invalid_taxonomy_value", f"$.taxonomy.{axis}", f"invalid `{axis}` value `{value}`", value))

    provenance = _as_mapping(candidate, "provenance")
    if not provenance.get("source_pool"):
        errors.append(_finding("missing_provenance", "$.provenance.source_pool", "source_pool is required"))
    if not provenance.get("source_id"):
        errors.append(_finding("missing_provenance", "$.provenance.source_id", "source_id is required"))
    source_artifacts = provenance.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        errors.append(_finding("missing_source_artifact", "$.provenance.source_artifacts", "at least one source artifact pointer is required"))

    expected_behavior = _as_mapping(candidate, "expected_behavior")
    for key in ("must", "must_not", "success_criteria"):
        if not isinstance(expected_behavior.get(key), list) or not expected_behavior.get(key):
            errors.append(_finding("missing_expected_behavior", f"$.expected_behavior.{key}", f"expected_behavior.{key} must be a non-empty list"))

    privacy_notes = _as_mapping(candidate, "privacy_notes")
    privacy_class = taxonomy.get("privacy_class")
    if privacy_notes.get("classification") != privacy_class:
        errors.append(_finding("privacy_class_mismatch", "$.privacy_notes.classification", "privacy_notes.classification must match taxonomy.privacy_class", privacy_notes.get("classification")))
    if privacy_notes.get("raw_payloads_included") is True:
        errors.append(_finding("raw_payload_declared", "$.privacy_notes.raw_payloads_included", "candidate declares raw payload inclusion"))
    if privacy_notes.get("private_identifiers_included") is True:
        errors.append(_finding("private_identifier_declared", "$.privacy_notes.private_identifiers_included", "candidate declares private identifier inclusion"))
    if privacy_notes.get("redactions_required") is True or privacy_class == "sensitive_requires_redaction":
        warnings.append(_finding("redaction_required", "$.privacy_notes.redactions_required", "candidate requires redaction before promotion"))

    risk = _as_mapping(candidate, "risk")
    side_effect_risk = taxonomy.get("side_effect_risk")
    if risk.get("side_effect_risk") not in (None, side_effect_risk):
        errors.append(_finding("side_effect_risk_mismatch", "$.risk.side_effect_risk", "risk.side_effect_risk must match taxonomy.side_effect_risk", risk.get("side_effect_risk")))
    if side_effect_risk == "external_write_requires_approval" and risk.get("external_write_approved") is not True:
        errors.append(_finding("unapproved_external_write", "$.risk.external_write_approved", "external-write candidate lacks explicit approval"))
    if side_effect_risk == "destructive_or_credential_change_prohibited" or risk.get("destructive_or_credential_change") is True:
        errors.append(_finding("prohibited_side_effect", "$.risk.side_effect_risk", "destructive or credential-change candidates are prohibited"))

    review_evidence = _as_mapping(candidate, "review_evidence")
    if taxonomy.get("regression_tier") == "blocking_regression":
        if review_evidence.get("reviewer_approved") is not True:
            errors.append(_finding("blocking_without_reviewer_approval", "$.review_evidence.reviewer_approved", "blocking regression requires reviewer approval"))
        if not _has_proof_artifact(review_evidence):
            errors.append(_finding("blocking_without_proof_artifact", "$.review_evidence.proof_artifacts", "blocking regression requires proof artifact path and hash"))
        if taxonomy.get("score_reliability") not in {"deterministic", "artifact_evidence_backed"}:
            errors.append(_finding("blocking_without_reliable_score", "$.taxonomy.score_reliability", "blocking regression requires deterministic or artifact-backed score reliability", taxonomy.get("score_reliability")))

    findings = _privacy_findings(candidate)
    raw_payloads_detected = any(finding["kind"] in {"raw_payload_marker"} for finding in findings)
    secret_like_values_detected = any(finding["kind"] == "secret_like_value" for finding in findings)
    private_identifier_markers_detected = any(finding["kind"] == "private_identifier_marker" for finding in findings)
    if raw_payloads_detected:
        errors.append(_finding("raw_payload_detected", "$", "raw payload markers are prohibited"))
    if secret_like_values_detected:
        errors.append(_finding("secret_like_value_detected", "$", "secret-like values are prohibited"))
    if private_identifier_markers_detected:
        errors.append(_finding("private_identifier_detected", "$", "private identifier markers are prohibited"))
    if privacy_class == "prohibited_raw_or_secret":
        errors.append(_finding("prohibited_privacy_class", "$.taxonomy.privacy_class", "prohibited_raw_or_secret candidates cannot be promoted"))

    status = "reject" if errors else ("needs_redaction" if warnings else "pass")
    return {
        "candidate_id": candidate_id,
        "schema": SCHEMA,
        "status": status,
        "privacy_screen": {
            "raw_payloads_detected": raw_payloads_detected,
            "secret_like_values_detected": secret_like_values_detected,
            "private_identifier_markers_detected": private_identifier_markers_detected,
            "redaction_required": any(warning["kind"] == "redaction_required" for warning in warnings),
        },
        "errors": errors,
        "warnings": warnings,
        "findings": findings,
    }


def build_validation_report(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results = [validate_candidate(candidate) for candidate in candidates]
    summary = {
        "candidate_count": len(results),
        "passed": sum(1 for result in results if result["status"] == "pass"),
        "needs_redaction": sum(1 for result in results if result["status"] == "needs_redaction"),
        "rejected": sum(1 for result in results if result["status"] == "reject"),
    }
    return {
        "schema": SCHEMA,
        "mode": MODE,
        "write_enabled": False,
        "langfuse_write_path_present": False,
        "summary": summary,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LF4 candidate metadata schema/privacy screen without Langfuse writes.")
    parser.add_argument("--candidates-json", required=True, type=Path, help="Candidate JSON object/list or envelope with candidates")
    parser.add_argument("--output-json", type=Path, help="Optional path for validation report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        candidates = _unwrap_candidates(_load_json(args.candidates_json))
        report = build_validation_report(candidates)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    except CandidateMetadataError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
