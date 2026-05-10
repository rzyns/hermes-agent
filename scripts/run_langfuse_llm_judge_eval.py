#!/usr/bin/env python3
"""Run local no-write LF5 LLM-as-judge calibration scaffolding.

This runner consumes only local LF5/Phase3/LF4 artifacts and emits sanitized
JSON/Markdown outputs. It never calls Langfuse APIs and has no live-write mode.
The default judge is deterministic and mockable; future provider-backed judging
must plug in by supplying the same in-memory Judge interface and must keep all
outputs advisory, local, and secret/raw-payload safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

DEFAULT_ARTIFACT_ROOT = Path("/home/openclaw/.hermes/artifacts/hermes-agent/langfuse-quality-llm-judge")
DEFAULT_RUBRIC_JSON = DEFAULT_ARTIFACT_ROOT / "lf5-00-llm-judge-score-taxonomy-rubric-packet-2026-05-10.json"
DEFAULT_REVIEW_JSON = DEFAULT_ARTIFACT_ROOT / "lf5-01-independent-review-lf5-00-rubric-boundaries-2026-05-10.json"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_ROOT
RUNNER_SCHEMA = "lf5_10_no_write_llm_judge_eval_v1"

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("langfuse_secret_key", re.compile(r"sk-lf-[A-Za-z0-9_-]+")),
    ("langfuse_public_key", re.compile(r"pk-lf-[A-Za-z0-9_-]+")),
    ("authorization_bearer", re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]{8,}=*")),
    ("generic_assignment_secret", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,'\"}]+")),
)
RAW_PAYLOAD_KEYS = {
    "input",
    "output",
    "messages",
    "candidate_output",
    "candidate_outputs",
    "candidate_text",
    "trace_payload",
    "tool_payload",
    "tool_args",
    "tool_output",
    "transcript",
    "raw_trace",
    "raw_payload",
}
ALLOWED_PRIVACY_CHECKS = {"no_raw_payloads_or_secrets_copied", "possible_risk_needs_review", "risk_observed"}


class LLMJudgeRunnerError(ValueError):
    """Raised when LF5 no-write judge evaluation cannot proceed safely."""


class Judge(Protocol):
    """Mockable judge interface for local/advisory LF5 calibration."""

    name: str

    def judge(self, fixture: Mapping[str, Any], rubric: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Return one advisory judge result for a fixture."""


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


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_id(value: Any) -> str:
    return str(value or "").strip()


def _score_names(rubric: Mapping[str, Any]) -> list[str]:
    scores = rubric.get("scores")
    if not isinstance(scores, list):
        raise LLMJudgeRunnerError("rubric scores must be a list")
    names: list[str] = []
    for score in scores:
        if not isinstance(score, Mapping) or not score.get("name"):
            raise LLMJudgeRunnerError("every rubric score must have a name")
        names.append(str(score["name"]))
    return names


def _score_specs(rubric: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(score["name"]): score for score in rubric.get("scores", []) if isinstance(score, Mapping) and score.get("name")}


def _scan_secret_findings(value: Any, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lower_key = str(key).lower()
            if any(part in lower_key for part in ("api_key", "secret", "token", "password", "authorization", "private_key")):
                # Allow scanner bookkeeping counters, but not sensitive value paths.
                if lower_key not in {"secret_like_findings_count", "secret_scan_status"}:
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
            findings.append({"path": path, "pattern": name, "preview": "[REDACTED]"})
    return findings


def _assert_no_raw_payload_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in RAW_PAYLOAD_KEYS:
                raise LLMJudgeRunnerError(f"raw payload field is not allowed in runner output: {child_path}")
            _assert_no_raw_payload_keys(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_no_raw_payload_keys(child, f"{path}[{idx}]")


def _assert_no_write_policy(rubric: Mapping[str, Any], review: Mapping[str, Any] | None = None) -> None:
    policy = rubric.get("policy") if isinstance(rubric.get("policy"), Mapping) else {}
    target = rubric.get("target_type_decision") if isinstance(rubric.get("target_type_decision"), Mapping) else {}
    forbidden = {str(item).lower() for item in rubric.get("forbidden_actions", []) if item}
    if policy.get("may_not_write_to_langfuse_without_separate_human_approval") is not True:
        raise LLMJudgeRunnerError("rubric does not fail closed on Langfuse writes")
    if policy.get("advisory_only") is not True or policy.get("non_blocking") is not True:
        raise LLMJudgeRunnerError("rubric must be advisory-only and non-blocking")
    if target.get("write_target_in_lf5") != "none_local_json_only":
        raise LLMJudgeRunnerError("rubric write_target_in_lf5 must be none_local_json_only")
    if not any("langfuse" in item and "write" in item for item in forbidden):
        raise LLMJudgeRunnerError("rubric forbidden_actions must include Langfuse writes")
    if review:
        verdicts = [review.get("mechanical_verdict"), review.get("substantive_verdict"), review.get("overall_verdict")]
        if any(str(verdict) != "PASS" for verdict in verdicts if verdict is not None):
            raise LLMJudgeRunnerError("LF5-01 review is not PASS/SUPPORTED")


def _artifact_pointer(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value))
    exists = path.exists()
    pointer: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists and path.is_file():
        pointer.update({"sha256": _sha256_file(path), "size_bytes": path.stat().st_size})
    return pointer


def _build_evidence_index(rubric: Mapping[str, Any]) -> dict[str, Any]:
    source_artifacts = rubric.get("source_artifacts") if isinstance(rubric.get("source_artifacts"), list) else []
    pointers = [_artifact_pointer(item) for item in source_artifacts]
    return {
        "source_artifacts": pointers,
        "missing_source_artifact_count": sum(1 for item in pointers if not item["exists"]),
        "evidence_policy": "minimized_artifact_pointers_only_no_raw_payloads",
    }


def _prompt_packet_for_fixture(fixture: Mapping[str, Any], rubric: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    prompt_sketch = rubric.get("judge_prompt_sketch") if isinstance(rubric.get("judge_prompt_sketch"), Mapping) else {}
    return {
        "fixture_id": _safe_id(fixture.get("id")),
        "system": prompt_sketch.get("system", ""),
        "input": {
            "fixture_id": _safe_id(fixture.get("id")),
            "fixture_expected_behavior": str(fixture.get("expected") or ""),
            "fixture_expected_negative": bool(fixture.get("expected_negative")),
            "fixture_purpose": str(fixture.get("purpose") or ""),
            "allowed_evidence_pointers": [item["path"] for item in evidence.get("source_artifacts", []) if item.get("exists")],
            "forbidden_actions": list(rubric.get("forbidden_actions", [])) if isinstance(rubric.get("forbidden_actions"), list) else [],
            "rubric_version": rubric.get("schema"),
        },
        "required_output_shape": prompt_sketch.get("required_output_shape", {}),
        "privacy_note": "Do not copy raw trace/tool payloads, transcripts, env values, or credentials.",
    }


class DeterministicScaffoldJudge:
    """Deterministic local judge used until an explicitly authorized provider is plugged in."""

    name = "deterministic_scaffold_v1"

    def judge(self, fixture: Mapping[str, Any], rubric: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
        fixture_id = _safe_id(fixture.get("id"))
        expected_negative = bool(fixture.get("expected_negative"))
        expected_text = str(fixture.get("expected") or "")
        missing_evidence = int(evidence.get("missing_source_artifact_count") or 0)
        raw_required = "raw payload" in expected_text.lower() and "forbidden" not in expected_text.lower()
        judge_abstained = missing_evidence > 0 or raw_required
        abstain_reason = None
        if missing_evidence > 0:
            abstain_reason = "insufficient_minimized_evidence"
        elif raw_required:
            abstain_reason = "raw_payload_required_but_forbidden"
        scores: dict[str, Any] = {
            "judge_groundedness": "abstain_insufficient_evidence" if judge_abstained else "supported",
            "judge_completeness": "abstain_not_applicable" if judge_abstained else "complete",
            "judge_instruction_following": "abstain_conflicting_or_missing_instructions" if judge_abstained else "followed",
            "judge_tool_use_appropriateness": "abstain_no_tool_context" if judge_abstained else "appropriate",
            "judge_privacy_risk": "abstain_raw_payload_not_available" if raw_required else "no_risk_observed",
            "judge_expected_negative_preservation": 1 if expected_negative and not judge_abstained else "abstain_not_expected_negative_fixture",
            "judge_abstained": 1 if judge_abstained else 0,
        }
        return {
            "fixture_id": fixture_id,
            "target": {"type": "dataset_run_item", "id": fixture_id, "write_target": "none_local_json_only"},
            "judge_name": self.name,
            "judge_abstained": judge_abstained,
            "abstain_reason": abstain_reason,
            "expected_negative": expected_negative,
            "expected_negative_preserved": expected_negative and not judge_abstained,
            "privacy_check": "no_raw_payloads_or_secrets_copied" if not raw_required else "possible_risk_needs_review",
            "scores": scores,
            "rationale": "Deterministic LF5 scaffold evaluated minimized fixture metadata and artifact pointers only; no live writes or raw payloads used.",
            "evidence_pointers": [item["path"] for item in evidence.get("source_artifacts", []) if item.get("exists")],
            "non_claims": [
                "advisory_local_result_only",
                "not_a_live_langfuse_score",
                "not_a_blocking_gate",
                "not_a_judge_accuracy_claim",
            ],
        }


def _validate_judge_result(result: Mapping[str, Any], rubric: Mapping[str, Any]) -> None:
    required = {"fixture_id", "judge_abstained", "abstain_reason", "privacy_check", "scores", "rationale", "non_claims"}
    missing = sorted(required - set(result.keys()))
    if missing:
        raise LLMJudgeRunnerError(f"judge result missing required fields: {', '.join(missing)}")
    if result.get("privacy_check") not in ALLOWED_PRIVACY_CHECKS:
        raise LLMJudgeRunnerError("judge result has invalid privacy_check")
    if not isinstance(result.get("scores"), Mapping):
        raise LLMJudgeRunnerError("judge result scores must be an object")
    expected_names = set(_score_names(rubric))
    actual_names = set(result["scores"].keys())
    if actual_names != expected_names:
        raise LLMJudgeRunnerError(f"judge result score names mismatch: expected {sorted(expected_names)}, got {sorted(actual_names)}")
    specs = _score_specs(rubric)
    for name, value in result["scores"].items():
        spec = specs[name]
        allowed = spec.get("range_or_categories") if isinstance(spec.get("range_or_categories"), list) else []
        if value not in allowed:
            raise LLMJudgeRunnerError(f"judge result score {name} has invalid value {value!r}")
    target = result.get("target") if isinstance(result.get("target"), Mapping) else {}
    if target.get("write_target") != "none_local_json_only":
        raise LLMJudgeRunnerError("judge result target must remain none_local_json_only")


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_negative_total = sum(1 for item in results if item.get("expected_negative") is True)
    expected_negative_preserved = sum(1 for item in results if item.get("expected_negative_preserved") is True)
    abstain_count = sum(1 for item in results if item.get("judge_abstained") is True)
    privacy_counts: dict[str, int] = {}
    for item in results:
        privacy_counts[str(item.get("privacy_check"))] = privacy_counts.get(str(item.get("privacy_check")), 0) + 1
    return {
        "fixture_count": len(results),
        "abstain_count": abstain_count,
        "expected_negative_fixture_count": expected_negative_total,
        "expected_negative_preserved_count": expected_negative_preserved,
        # Keep enum values in values, not JSON key paths, so conservative secret
        # scanners do not flag the sanctioned no_raw_payloads_or_secrets_copied label.
        "privacy_check_summary": [
            {"category": category, "count": count}
            for category, count in sorted(privacy_counts.items())
        ],
        "write_enabled": False,
        "langfuse_api_calls_attempted": False,
    }


def build_llm_judge_eval(
    rubric_json: Path,
    output_dir: Path,
    *,
    review_json: Path | None = None,
    judge: Judge | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    rubric = _load_json(rubric_json)
    if not isinstance(rubric, Mapping):
        raise LLMJudgeRunnerError("rubric JSON must be an object")
    review = None
    if review_json and review_json.exists():
        review_payload = _load_json(review_json)
        if not isinstance(review_payload, Mapping):
            raise LLMJudgeRunnerError("review JSON must be an object")
        review = review_payload
    _assert_no_write_policy(rubric, review)
    fixtures = rubric.get("calibration_fixtures")
    if not isinstance(fixtures, list) or not all(isinstance(item, Mapping) for item in fixtures):
        raise LLMJudgeRunnerError("rubric calibration_fixtures must be a list of objects")
    evidence = _build_evidence_index(rubric)
    selected_judge = judge or DeterministicScaffoldJudge()
    run_id = run_id or f"lf5-10-llm-judge-eval-{_utc_stamp()}"

    prompt_packets = [_prompt_packet_for_fixture(fixture, rubric, evidence) for fixture in fixtures]
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        result = selected_judge.judge(fixture, rubric, evidence)
        _validate_judge_result(result, rubric)
        results.append(result)

    report: dict[str, Any] = {
        "schema": RUNNER_SCHEMA,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "local_no_write_llm_judge_eval_scaffold",
        "advisory_only": True,
        "write_enabled": False,
        "langfuse_api_calls_attempted": False,
        "langfuse_writes_attempted": False,
        "scheduler_mutations_attempted": False,
        "blocking_gate_promotion_attempted": False,
        "public_postback_attempted": False,
        "judge": {"name": selected_judge.name, "provider": "local_deterministic_or_injected_mock", "live_provider_invoked": False},
        "source": {
            "rubric_json": str(rubric_json),
            "rubric_sha256": _sha256_file(rubric_json),
            "review_json": str(review_json) if review_json else None,
            "review_sha256": _sha256_file(review_json) if review_json and review_json.exists() else None,
        },
        "evidence_index": evidence,
        "aggregate": _aggregate(results),
        "results": results,
        "prompt_packets": prompt_packets,
        "non_claims": [
            "No Langfuse API calls, dataset writes, run writes, or score writes were attempted.",
            "No scheduler/cron mutation, public postback, blocking gate promotion, deploy, push, merge, or restart was attempted.",
            "Deterministic scaffold output does not prove judge accuracy; LF5-11/12/13 must calibrate/review before LF5-20 recommendation.",
            "Artifacts contain minimized pointers and fixture metadata only, not raw trace/tool payloads.",
        ],
    }
    _assert_no_raw_payload_keys({"report": {key: value for key, value in report.items() if key != "prompt_packets"}})
    # prompt_packets intentionally contain a minimized `input` object because LF5-00
    # names that prompt field; scan it for secrets separately rather than treating it
    # as a persisted raw trace/tool payload.
    secret_findings = _scan_secret_findings(report)
    report["secret_scan"] = {"status": "passed" if not secret_findings else "failed", "secret_like_findings_count": len(secret_findings), "findings": secret_findings}
    if secret_findings:
        raise LLMJudgeRunnerError("runner report contains secret-like findings")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"
    md_path = output_dir / f"{run_id}.md"
    _write_json(json_path, report)
    md_path.write_text(_markdown_report(report, json_path, md_path))
    report["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    _write_json(json_path, report)
    return report


def _markdown_report(report: Mapping[str, Any], json_path: Path, md_path: Path) -> str:
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), Mapping) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), Mapping) else {}
    lines = [
        "# LF5-10 no-write LLM-as-judge evaluation scaffold",
        "",
        f"Run id: `{report.get('run_id')}`",
        f"Schema: `{report.get('schema')}`",
        "",
        "## Guardrails",
        "",
        "- write_enabled: false",
        "- langfuse_api_calls_attempted: false",
        "- langfuse_writes_attempted: false",
        "- scheduler_mutations_attempted: false",
        "- blocking_gate_promotion_attempted: false",
        "- public_postback_attempted: false",
        "",
        "## Aggregate",
        "",
        f"- fixtures: {aggregate.get('fixture_count')}",
        f"- abstains: {aggregate.get('abstain_count')}",
        f"- expected-negative fixtures: {aggregate.get('expected_negative_fixture_count')}",
        f"- expected-negative preserved: {aggregate.get('expected_negative_preserved_count')}",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{artifacts.get('json', str(json_path))}`",
        f"- Markdown: `{artifacts.get('markdown', str(md_path))}`",
        "",
        "## Known limitations",
        "",
        "- Default judging is deterministic scaffolding, not calibrated LLM judgment.",
        "- Provider-backed judging is deliberately an injectable interface and remains no-write/local-only unless separately authorized.",
        "- LF5-11/12/13 must run calibration, comparison, and independent audit before any LF5-20 recommendation.",
        "",
        "## Next LF5-11 run command",
        "",
        "```bash",
        f"python scripts/run_langfuse_llm_judge_eval.py --rubric-json {report.get('source', {}).get('rubric_json')} --review-json {report.get('source', {}).get('review_json')} --output-dir {DEFAULT_OUTPUT_DIR}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local no-write LF5 LLM-as-judge evaluation scaffolding")
    parser.add_argument("--rubric-json", type=Path, default=DEFAULT_RUBRIC_JSON)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    report = build_llm_judge_eval(args.rubric_json, args.output_dir, review_json=args.review_json, run_id=args.run_id)
    stdout = {
        "schema": report["schema"],
        "run_id": report["run_id"],
        "write_enabled": report["write_enabled"],
        "langfuse_api_calls_attempted": report["langfuse_api_calls_attempted"],
        "langfuse_writes_attempted": report["langfuse_writes_attempted"],
        "aggregate": report["aggregate"],
        "artifacts": report.get("artifacts", {}),
    }
    print(json.dumps(stdout, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
