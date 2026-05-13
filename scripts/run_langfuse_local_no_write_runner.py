#!/usr/bin/env python3
"""Run the LF13 local/no-write Langfuse dataset/eval artifact slice.

This is an explicit-source, artifact-only runner. It does not query Langfuse,
does not write Langfuse datasets/runs/scores, and refuses broad trace filters or
raw prompt/output/tool payload persistence. The implementation intentionally
wraps the existing local candidate-output, local eval, and human-review helpers
instead of creating a parallel evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical_key(value: str) -> str:
    """Return a case/separator-insensitive key form for fail-closed privacy matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)

BROAD_SOURCE_KEYS = {
    "trace_filter",
    "trace_filters",
    "tracefilter",
    "session_filter",
    "session_filters",
    "sessionfilter",
    "query",
    "search_query",
    "searchquery",
    "where",
    "from_timestamp",
    "fromtimestamp",
    "to_timestamp",
    "totimestamp",
    "broad_trace_filters",
    "broadtracefilters",
    "allow_broad_trace_filters",
    "allowbroadtracefilters",
}

BROAD_SOURCE_KEYS_CANONICAL = frozenset(_canonical_key(key) for key in BROAD_SOURCE_KEYS)

SECRET_KEY_NAMES_CANONICAL = frozenset(
    _canonical_key(key)
    for key in {
        "api_key",
        "apikey",
        "secret",
        "token",
        "password",
        "passwd",
        "authorization",
        "bearer",
        "langfuse_secret_key",
        "langfuse_public_key",
    }
)

RAW_PAYLOAD_KEYS = {
    "raw_prompt",
    "raw_prompts",
    "raw_input",
    "raw_inputs",
    "raw_output",
    "raw_outputs",
    "raw_messages",
    "messages",
    "input",
    "output",
    "tool_payload",
    "tool_payloads",
    "tool_args",
    "tool_arguments",
    "tool_output",
    "tool_outputs",
    "trace_payload",
    "observation_payload",
}

RAW_PAYLOAD_KEYS_CANONICAL = frozenset(_canonical_key(key) for key in RAW_PAYLOAD_KEYS)

SCRIPT_DIR = Path(__file__).resolve().parent
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


class LF13NoWriteRunnerError(ValueError):
    """Raised when the LF13 runner cannot safely produce no-write artifacts."""


def _require_safe_identifier(value: Any, *, path: str, required: bool = True) -> str:
    if value is None or value == "":
        if required:
            raise LF13NoWriteRunnerError(f"identifier field is required: {path}")
        return ""
    if not isinstance(value, str):
        raise LF13NoWriteRunnerError(f"identifier field must be a string: {path}")
    text = value
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(text):
        raise LF13NoWriteRunnerError(f"identifier field contains unsafe characters: {path}")
    return text


def _load_peer_module(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive import guard
        raise LF13NoWriteRunnerError(f"cannot load helper module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_outputs_module() -> Any:
    return _load_peer_module("generate_langfuse_candidate_outputs.py", "lf13_generate_langfuse_candidate_outputs")


def _dataset_eval_module() -> Any:
    return _load_peer_module("run_langfuse_dataset_eval.py", "lf13_run_langfuse_dataset_eval")


def _review_packet_module() -> Any:
    return _load_peer_module("generate_langfuse_human_review_packet.py", "lf13_generate_langfuse_human_review_packet")


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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scan_value(value: Any, *, path: str = "", scan_raw_keys: bool = True) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_normalized = _canonical_key(key_text)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_normalized in SECRET_KEY_NAMES_CANONICAL:
                findings.append({"path": child_path, "pattern": "secret_key_name", "preview": "[REDACTED]"})
            if scan_raw_keys and key_normalized in RAW_PAYLOAD_KEYS_CANONICAL:
                findings.append({"path": child_path, "pattern": "raw_payload_key", "preview": "[REDACTED]"})
            findings.extend(_scan_value(child, path=child_path, scan_raw_keys=scan_raw_keys))
        return findings
    if isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(_scan_value(child, path=f"{path}[{idx}]", scan_raw_keys=scan_raw_keys))
        return findings
    if value is None or isinstance(value, (bool, int, float)):
        return findings
    text = str(value)
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append({"path": path, "pattern": name, "preview": redact_text(text)[:160]})
    return findings


def _scan_broad_source_keys(value: Any, *, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_normalized = _canonical_key(key_text)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_normalized in BROAD_SOURCE_KEYS_CANONICAL and not (key_normalized == "allowbroadtracefilters" and child is False):
                findings.append({"path": child_path, "pattern": "broad_trace_filter", "preview": "[REDACTED]"})
            findings.extend(_scan_broad_source_keys(child, path=child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(_scan_broad_source_keys(child, path=f"{path}[{idx}]"))
    return findings


def _validate_seed(seed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if seed.get("write_enabled") is not False:
        raise LF13NoWriteRunnerError("seed must set write_enabled=false")
    source_policy = seed.get("source_policy") if isinstance(seed.get("source_policy"), Mapping) else {}
    explicit_sources = seed.get("explicit_sources") if isinstance(seed.get("explicit_sources"), Mapping) else {}
    if source_policy.get("allow_broad_trace_filters") is not False or explicit_sources.get("allow_broad_trace_filters"):
        raise LF13NoWriteRunnerError("broad trace filters are not allowed in LF13 local/no-write mode")
    broad_findings = _scan_broad_source_keys(seed)
    if broad_findings:
        raise LF13NoWriteRunnerError(f"broad trace filters are not allowed in LF13 local/no-write mode: {json.dumps(broad_findings, sort_keys=True)}")
    privacy_findings = _scan_value(seed)
    if privacy_findings:
        raise LF13NoWriteRunnerError(f"privacy screen failed for seed: {json.dumps(privacy_findings, sort_keys=True)}")

    raw_trace_ids = explicit_sources.get("trace_ids", [])
    if not isinstance(raw_trace_ids, list) or not all(isinstance(item, str) for item in raw_trace_ids):
        raise LF13NoWriteRunnerError("seed explicit_sources.trace_ids must be a list of strings")
    allowed_trace_ids = {
        _require_safe_identifier(item, path=f"explicit_sources.trace_ids[{index}]")
        for index, item in enumerate(raw_trace_ids)
    }
    candidates = explicit_sources.get("candidates")
    if not isinstance(candidates, list) or not all(isinstance(item, Mapping) for item in candidates):
        raise LF13NoWriteRunnerError("seed explicit_sources.candidates must be a list of objects")
    if not candidates:
        raise LF13NoWriteRunnerError("at least one explicit candidate is required")

    for candidate_index, candidate in enumerate(candidates):
        candidate_path = f"explicit_sources.candidates[{candidate_index}]"
        trace_id = _require_safe_identifier(candidate.get("source_trace_id"), path=f"{candidate_path}.source_trace_id")
        _require_safe_identifier(candidate.get("dataset_item_id"), path=f"{candidate_path}.dataset_item_id")
        _require_safe_identifier(candidate.get("session_id"), path=f"{candidate_path}.session_id", required=False)
        _require_safe_identifier(candidate.get("turn_id"), path=f"{candidate_path}.turn_id", required=False)
        promotion_reason = candidate.get("promotion_reason")
        _require_safe_identifier(
            "explicit_reviewed_source" if promotion_reason is None or promotion_reason == "" else promotion_reason,
            path=f"{candidate_path}.promotion_reason",
        )
        if allowed_trace_ids and trace_id not in allowed_trace_ids:
            raise LF13NoWriteRunnerError("candidate source_trace_id is outside explicit trace allowlist")
    return candidates


def _safe_nonnegative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LF13NoWriteRunnerError(f"summary field must be a non-negative integer: {path} ({type(value).__name__})")
    if value < 0:
        raise LF13NoWriteRunnerError(f"summary field must be a non-negative integer: {path} (negative)")
    return value


def _safe_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    summary = candidate.get("summary") if isinstance(candidate.get("summary"), Mapping) else {}
    allowed_count_keys = {
        "tool_observations",
        "tool_null_output_count",
        "tool_outputs_present_count",
        "tool_call_id_present_count",
        "tool_args_present_count",
        "score_count",
    }
    out = {}
    for key in allowed_count_keys:
        if key in summary:
            out[key] = _safe_nonnegative_int(summary[key], path=f"summary.{key}")
    out["trace_id"] = str(candidate.get("source_trace_id") or "")
    return out


def _artifact_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), Mapping) else {}
    summary = _safe_summary(candidate)
    default_null_zero = int(summary.get("tool_null_output_count", 0) or 0) == 0
    default_tool_outputs = int(summary.get("tool_outputs_present_count", 0) or 0) > 0
    default_ids_args = int(summary.get("tool_call_id_present_count", 0) or 0) > 0 and int(summary.get("tool_args_present_count", 0) or 0) > 0
    return {
        "summary": summary,
        "tool_outputs_present": bool(evidence.get("tool_outputs_present", default_tool_outputs)),
        "tool_call_ids_and_args_present": bool(evidence.get("tool_call_ids_and_args_present", default_ids_args)),
        "tool_null_outputs_zero": bool(evidence.get("tool_null_outputs_zero", default_null_zero)),
    }


def _build_candidate_queue(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queued: list[dict[str, Any]] = []
    for candidate in candidates:
        queued.append({
            "dataset_item_id": str(candidate.get("dataset_item_id") or ""),
            "source_trace_id": str(candidate.get("source_trace_id") or ""),
            "session_id": str(candidate.get("session_id") or ""),
            "turn_id": str(candidate.get("turn_id") or ""),
            "promotion_reason": str(candidate.get("promotion_reason") or "explicit_reviewed_source"),
            "decision": "accepted_for_local_no_write_fixture",
            "decision_reason": "explicit reviewed source only; no broad trace mining performed",
            "raw_payloads_copied": False,
        })
    return {
        "mode": "lf13_candidate_queue_no_write",
        "created_at_utc": _now(),
        "write_enabled": False,
        "explicit_source_only": True,
        "summary": {"candidate_count": len(queued), "accepted_count": len(queued), "deferred_count": 0, "rejected_count": 0},
        "candidates": queued,
    }


def _build_safe_trace_summaries(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        score_summary = candidate.get("score_summary") if isinstance(candidate.get("score_summary"), Mapping) else {}
        entry = {
            "source_trace_id": str(candidate.get("source_trace_id") or ""),
            "session_id": str(candidate.get("session_id") or ""),
            "turn_id": str(candidate.get("turn_id") or ""),
            "summary": _safe_summary(candidate),
            "evidence_flags": {key: value for key, value in _artifact_evidence(candidate).items() if isinstance(value, bool)},
            "raw_payloads_copied": False,
        }
        if score_summary:
            entry["score_summary"] = {
                "name": str(score_summary.get("name") or ""),
                "value": score_summary.get("value") if isinstance(score_summary.get("value"), (str, int, float, bool)) else None,
                "write_status": "not_written_local_proposal_only",
            }
        summaries.append(entry)
    return {
        "mode": "lf13_safe_trace_summaries_no_write",
        "created_at_utc": _now(),
        "write_enabled": False,
        "summary": {"trace_summary_count": len(summaries), "raw_payloads_copied": False},
        "traces": summaries,
    }


def _build_fixture(candidate: Mapping[str, Any]) -> dict[str, Any]:
    dataset_item_id = str(candidate.get("dataset_item_id") or "")
    trace_id = str(candidate.get("source_trace_id") or "")
    return {
        "schema": "hermes.eval.fixture.v1",
        "dataset_item_id": dataset_item_id,
        "provenance": {
            "source": "explicit_reviewed_langfuse_trace",
            "source_trace_id": trace_id,
            "session_id": str(candidate.get("session_id") or ""),
            "turn_id": str(candidate.get("turn_id") or ""),
            "promotion_reason": str(candidate.get("promotion_reason") or "explicit_reviewed_source"),
        },
        "sanitized_task_spec": {
            "kind": "trace_quality_summary",
            "description": "Assess a reviewed Hermes/Langfuse turn using minimized evidence only.",
            "raw_payloads_available": False,
        },
        "expected_behavior": {
            "must": [
                "state that the evaluation is local/no-write",
                "use only minimized trace/tool evidence",
                "leave semantic judgment for manual review when evidence is insufficient",
            ],
            "must_not": [
                "include credentials or private values",
                "persist raw prompt, output, message, trace, or tool payloads",
                "claim a live Langfuse dataset/run/score write occurred",
            ],
        },
        "evaluation_contract": {
            "checks": [
                {"name": "privacy_safe", "type": "secret_scan", "target": "candidate_output"},
                {"name": "tool_outputs_present", "type": "deterministic_artifact_check", "target": "artifact_evidence"},
                {"name": "tool_call_ids_and_args_present", "type": "deterministic_artifact_check", "target": "artifact_evidence"},
                {"name": "tool_null_outputs_zero", "type": "deterministic_artifact_check", "target": "artifact_evidence"},
                {"name": "semantic_grounding", "type": "manual_review", "target": "candidate_output"},
            ]
        },
        "review": {"status": "manual_review_pending", "notes": []},
    }


def _build_plan(fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    for fixture in fixtures:
        expected = fixture.get("expected_behavior") if isinstance(fixture.get("expected_behavior"), Mapping) else {}
        eval_contract = fixture.get("evaluation_contract") if isinstance(fixture.get("evaluation_contract"), Mapping) else {}
        provenance = fixture.get("provenance") if isinstance(fixture.get("provenance"), Mapping) else {}
        contracts.append({
            "dataset_item_id": fixture.get("dataset_item_id"),
            "source_trace_id": provenance.get("source_trace_id"),
            "status": "ACTIVE",
            "promotion_reason": provenance.get("promotion_reason"),
            "must": expected.get("must", []),
            "must_not": expected.get("must_not", []),
            "checks": [check.get("name") for check in eval_contract.get("checks", []) if isinstance(check, Mapping)],
            "deterministic_checks": eval_contract.get("checks", []),
        })
    return {
        "mode": "lf13_local_no_write_eval_plan",
        "dataset_name": "hermes/lf13-local-no-write-fixtures",
        "write_enabled": False,
        "summary": {"dataset_item_count": len(contracts)},
        "proposed_experiment": {
            "write_enabled": False,
            "requires_explicit_future_flags": ["future-live-write-gate", "human-approval"],
            "scoring_policy": "deterministic_and_manual_first",
        },
        "contracts": contracts,
    }


def _render_dry_run_markdown(dry_run: Mapping[str, Any]) -> str:
    summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), Mapping) else {}
    lines = [
        "# LF13 Local/No-Write Dry Run Results",
        "",
        f"Mode: `{dry_run.get('mode', '')}`",
        f"Write enabled: `{dry_run.get('write_enabled')}`",
        f"Langfuse writes attempted: `{dry_run.get('langfuse_writes_attempted')}`",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    lines.extend(["", "## Future approvals still blocked", ""])
    for approval in dry_run.get("blocked_future_approvals", []):
        lines.append(f"- {approval}")
    lines.extend(["", "## Item statuses", ""])
    for item in dry_run.get("items", []):
        if isinstance(item, Mapping):
            lines.append(f"- `{item.get('dataset_item_id')}` / `{item.get('source_trace_id')}`: `{item.get('status')}`")
    return "\n".join(lines).rstrip() + "\n"


def _privacy_screen(artifacts: Mapping[str, Path]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for name, path in artifacts.items():
        if not path.exists():
            findings.append({"artifact": name, "path": str(path), "pattern": "missing_artifact", "preview": "missing"})
            continue
        text = path.read_text()
        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"artifact": name, "path": str(path), "pattern": pattern_name, "preview": redact_text(text)[:160]})
        if path.suffix == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                findings.append({"artifact": name, "path": str(path), "pattern": "invalid_json", "preview": str(exc)[:160]})
                continue
            findings.extend({"artifact": name, **finding} for finding in _scan_value(payload, scan_raw_keys=True))
        elif path.suffix == ".jsonl":
            for idx, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    findings.append({"artifact": name, "path": f"{path}:{idx}", "pattern": "invalid_jsonl", "preview": str(exc)[:160]})
                    continue
                findings.extend({"artifact": name, **finding} for finding in _scan_value(payload, path=f"line[{idx}]", scan_raw_keys=True))
    return {
        "mode": "lf13_privacy_screen_no_write",
        "created_at_utc": _now(),
        "write_enabled": False,
        "status": "fail" if findings else "pass",
        "summary": {"artifact_count": len(artifacts), "finding_count": len(findings), "raw_payloads_copied": False},
        "findings": findings,
    }


def _privacy_screen_content(artifacts: Mapping[str, tuple[Path, str]]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for name, (path, text) in artifacts.items():
        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"artifact": name, "path": str(path), "pattern": pattern_name, "preview": redact_text(text)[:160]})
        if path.suffix == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                findings.append({"artifact": name, "path": str(path), "pattern": "invalid_json", "preview": str(exc)[:160]})
                continue
            findings.extend({"artifact": name, **finding} for finding in _scan_value(payload, scan_raw_keys=True))
        elif path.suffix == ".jsonl":
            for idx, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    findings.append({"artifact": name, "path": f"{path}:{idx}", "pattern": "invalid_jsonl", "preview": str(exc)[:160]})
                    continue
                findings.extend({"artifact": name, **finding} for finding in _scan_value(payload, path=f"line[{idx}]", scan_raw_keys=True))
    return {
        "mode": "lf13_privacy_screen_no_write",
        "created_at_utc": _now(),
        "write_enabled": False,
        "status": "fail" if findings else "pass",
        "summary": {"artifact_count": len(artifacts), "finding_count": len(findings), "raw_payloads_copied": False},
        "findings": findings,
    }


def _write_hash_manifest(path: Path, artifacts: Mapping[str, Path]) -> None:
    lines = []
    for artifact in sorted(artifacts.values(), key=lambda item: item.name):
        lines.append(f"{_sha256_file(artifact)}  {artifact.name}")
    path.write_text("\n".join(lines) + "\n")


def run_lf13_local_no_write(seed_json: Path, output_dir: Path) -> dict[str, Any]:
    seed = _load_json(seed_json)
    if not isinstance(seed, Mapping):
        raise LF13NoWriteRunnerError("seed JSON must be an object")
    candidates = _validate_seed(seed)

    candidate_queue = _build_candidate_queue(candidates)
    safe_summaries = _build_safe_trace_summaries(candidates)
    fixtures = [_build_fixture(candidate) for candidate in candidates]
    plan = _build_plan(fixtures)
    artifact_evidence = {
        str(candidate.get("dataset_item_id") or ""): _artifact_evidence(candidate)
        for candidate in candidates
    }
    artifact_evidence.update({
        str(candidate.get("source_trace_id") or ""): _artifact_evidence(candidate)
        for candidate in candidates
    })

    candidate_outputs_mod = _candidate_outputs_module()
    dataset_eval_mod = _dataset_eval_module()
    review_packet_mod = _review_packet_module()

    candidate_outputs = candidate_outputs_mod.build_candidate_outputs(plan, artifact_evidence=artifact_evidence)
    dry_run = dataset_eval_mod.build_local_eval_results(plan, candidate_outputs=candidate_outputs, artifact_evidence=artifact_evidence)
    dry_run = dict(dry_run)
    dry_run["mode"] = "lf13_local_no_write_dry_run_results"
    dry_run["write_enabled"] = False
    dry_run["langfuse_writes_attempted"] = False
    dry_run["candidate_decision_counts"] = candidate_queue["summary"]
    dry_run["privacy_decision_counts"] = {"pending_until_privacy_screen_written": True}
    dry_run["local_score_proposals_not_written"] = [
        {
            "dataset_item_id": str(candidate.get("dataset_item_id") or ""),
            "source_trace_id": str(candidate.get("source_trace_id") or ""),
            "name": str((candidate.get("score_summary") if isinstance(candidate.get("score_summary"), Mapping) else {}).get("name") or "local_quality_proposal"),
            "write_status": "not_written_local_proposal_only",
        }
        for candidate in candidates
    ]
    dry_run["blocked_future_approvals"] = [
        "live Langfuse dataset/run/score writes require a separate human-approved gate",
        "broad trace/session mining remains out of scope",
        "semantic checks remain manual-review placeholders",
    ]

    review_packet = review_packet_mod.build_review_packet(
        plan,
        eval_result=dry_run,
        candidate_outputs=candidate_outputs,
        artifact_evidence=artifact_evidence,
    )
    review_markdown = review_packet_mod.render_markdown(review_packet)

    artifacts: dict[str, Path] = {
        "candidate_queue": output_dir / "candidate_queue.json",
        "safe_trace_summaries": output_dir / "safe_trace_summaries.json",
        "dataset_fixture_candidates": output_dir / "dataset_fixture_candidates.jsonl",
        "dry_run_results": output_dir / "dry_run_results.json",
        "dry_run_markdown": output_dir / "dry_run_results.md",
        "review_packet": output_dir / "review_packet.json",
        "review_markdown": output_dir / "review_packet.md",
    }
    artifact_content: dict[str, tuple[Path, str]] = {
        "candidate_queue": (artifacts["candidate_queue"], json.dumps(candidate_queue, indent=2, sort_keys=True) + "\n"),
        "safe_trace_summaries": (artifacts["safe_trace_summaries"], json.dumps(safe_summaries, indent=2, sort_keys=True) + "\n"),
        "dataset_fixture_candidates": (artifacts["dataset_fixture_candidates"], "".join(json.dumps(item, sort_keys=True) + "\n" for item in fixtures)),
        "dry_run_results": (artifacts["dry_run_results"], json.dumps(dry_run, indent=2, sort_keys=True) + "\n"),
        "dry_run_markdown": (artifacts["dry_run_markdown"], _render_dry_run_markdown(dry_run)),
        "review_packet": (artifacts["review_packet"], json.dumps(review_packet, indent=2, sort_keys=True) + "\n"),
        "review_markdown": (artifacts["review_markdown"], review_markdown),
    }

    privacy_report = _privacy_screen_content(artifact_content)
    privacy_report["summary"]["candidate_count"] = len(candidates)
    if privacy_report["status"] != "pass":
        raise LF13NoWriteRunnerError(f"privacy screen failed: {json.dumps(privacy_report['findings'], sort_keys=True)}")

    dry_run["privacy_decision_counts"] = {
        "status": privacy_report["status"],
        "finding_count": privacy_report["summary"]["finding_count"],
    }
    artifacts["privacy_screen"] = output_dir / "privacy_screen_report.json"
    artifact_content["dry_run_results"] = (artifacts["dry_run_results"], json.dumps(dry_run, indent=2, sort_keys=True) + "\n")
    artifact_content["dry_run_markdown"] = (artifacts["dry_run_markdown"], _render_dry_run_markdown(dry_run))
    artifact_content["privacy_screen"] = (artifacts["privacy_screen"], json.dumps(privacy_report, indent=2, sort_keys=True) + "\n")
    final_privacy_report = _privacy_screen_content(artifact_content)
    if final_privacy_report["status"] != "pass":
        raise LF13NoWriteRunnerError(f"privacy screen failed: {json.dumps(final_privacy_report['findings'], sort_keys=True)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for _name, (path, content) in artifact_content.items():
        path.write_text(content)

    manifest_path = output_dir / "hash_manifest.sha256"
    _write_hash_manifest(manifest_path, artifacts)
    artifacts["hash_manifest"] = manifest_path

    return {
        "mode": "lf13_local_no_write_runner",
        "created_at_utc": _now(),
        "write_enabled": False,
        "langfuse_writes_attempted": False,
        "output_dir": str(output_dir),
        "summary": {
            "candidate_count": len(candidates),
            "fixture_count": len(fixtures),
            "artifact_count": len(artifacts),
            "privacy_status": privacy_report["status"],
            "manual_review_case_count": review_packet.get("summary", {}).get("case_count") if isinstance(review_packet.get("summary"), Mapping) else None,
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LF13 explicit-source local/no-write Langfuse dataset/eval artifact slice.")
    parser.add_argument("--seed-json", type=Path, required=True, help="Explicit-source seed JSON; broad filters are refused")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for local/no-write LF13 artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_lf13_local_no_write(args.seed_json, args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except LF13NoWriteRunnerError as exc:
        parser.error(redact_text(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
