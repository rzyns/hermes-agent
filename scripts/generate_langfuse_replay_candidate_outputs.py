#!/usr/bin/env python3
"""Generate tiny local replay/model candidate outputs for Langfuse eval review.

This helper is deliberately local/no-write. It builds prompts from a reviewed
plan plus minimized artifact evidence, runs a bounded Hermes one-shot model call
for a tiny selected subset, and emits evaluator-compatible candidate outputs.

Safety posture:
- no Langfuse API writes;
- child process Langfuse credential env is scrubbed;
- session saving is disabled when supported by Hermes;
- prompts use minimized evidence and manual-check names only, not raw trace/tool
  payloads and not copied contract `must`/`must_not` bullets.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence


LANGFUSE_ENV_KEYS = (
    "HERMES_LANGFUSE_PUBLIC_KEY",
    "HERMES_LANGFUSE_SECRET_KEY",
    "HERMES_LANGFUSE_BASE_URL",
    "HERMES_LANGFUSE_ENV",
    "HERMES_LANGFUSE_RELEASE",
    "HERMES_LANGFUSE_SAMPLE_RATE",
    "HERMES_LANGFUSE_MAX_CHARS",
    "HERMES_LANGFUSE_DEBUG",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LANGFUSE_ENV",
    "LANGFUSE_RELEASE",
)

DEFAULT_SOURCE_TRACE_IDS = (
    "0b7cf57223181f7cc413a01ea4c49c8f",  # canonical success
    "098ae0f47033aa7c9c301df17ac9f4e7",  # local default null-output failure
    "e76a953daa8b10e91d7743e5880dac59",  # privacy case with tool evidence
)
MAX_REPLAY_CASES = 30


class ReplayAdapterError(RuntimeError):
    """Raised when replay candidate generation cannot proceed safely."""


class ReplayRun(NamedTuple):
    exit_code: int
    stdout: str
    stderr: str


class CleanedCandidateOutput(NamedTuple):
    text: str
    runner_status_messages: list[str]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ReplayAdapterError(f"Invalid JSON in {path}: {exc}") from exc


def _unwrap_artifact_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("artifact_evidence", payload)
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def _contracts(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = plan.get("contracts")
    if not isinstance(raw, list):
        raise ReplayAdapterError("plan must contain a contracts list")
    return [contract for contract in raw if isinstance(contract, Mapping)]


def _manual_check_names(contract: Mapping[str, Any]) -> list[str]:
    checks = contract.get("deterministic_checks")
    if not isinstance(checks, list):
        return []
    names: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("type") == "manual_review":
            names.append(str(check.get("name") or "manual_review"))
    return names


def _deterministic_check_names(contract: Mapping[str, Any]) -> list[str]:
    checks = contract.get("deterministic_checks")
    if not isinstance(checks, list):
        return []
    names: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("type") != "manual_review":
            names.append(str(check.get("name") or "unknown_check"))
    return names


def _evidence_summary(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return {}
    summary = evidence.get("summary") if isinstance(evidence.get("summary"), Mapping) else evidence
    if not isinstance(summary, Mapping):
        return {}
    allowed = [
        "trace_id",
        "tool_observations",
        "tool_null_output_count",
        "tool_output_present_count",
        "tool_outputs_present_count",
        "tool_call_id_present_count",
        "tool_args_present_count",
    ]
    return {key: summary[key] for key in allowed if key in summary}


def _select_contracts(plan: Mapping[str, Any], source_trace_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    wanted = list(source_trace_ids)
    by_trace = {str(contract.get("source_trace_id") or ""): contract for contract in _contracts(plan)}
    selected: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for trace_id in wanted:
        contract = by_trace.get(trace_id)
        if contract is None:
            missing.append(trace_id)
        else:
            selected.append(contract)
    if missing:
        raise ReplayAdapterError(f"source trace ids not found in plan: {', '.join(missing)}")
    return selected


def _build_prompt(contract: Mapping[str, Any], evidence_summary: Mapping[str, Any]) -> str:
    dataset_item_id = str(contract.get("dataset_item_id") or "")
    source_trace_id = str(contract.get("source_trace_id") or "")
    promotion_reason = str(contract.get("promotion_reason") or "unknown")
    manual_checks = _manual_check_names(contract)
    deterministic_checks = _deterministic_check_names(contract)
    return "\n".join([
        "You are producing a local, no-write candidate output for a Langfuse evaluation dry run.",
        "Use only the minimized artifact evidence in this prompt. Do not invent raw trace content.",
        "Do not claim a final human-review pass; mark semantic judgments as needing hybrid human review when uncertain.",
        "Return concise prose with: Verdict, Evidence, Uncertainties, Suggested human-review question.",
        "",
        f"Dataset item id: {dataset_item_id}",
        f"Source trace id: {source_trace_id}",
        f"Promotion reason: {promotion_reason}",
        f"Manual semantic checks to address: {json.dumps(manual_checks, sort_keys=True)}",
        f"Automated/deterministic checks already tracked separately: {json.dumps(deterministic_checks, sort_keys=True)}",
        "Minimized evidence summary:",
        json.dumps(dict(evidence_summary), indent=2, sort_keys=True),
        "",
        "Important privacy boundary: this prompt intentionally omits raw trace inputs, raw tool inputs, and raw tool outputs.",
    ])


def build_replay_prompts(
    plan: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any],
    *,
    source_trace_ids: Sequence[str] = DEFAULT_SOURCE_TRACE_IDS,
) -> list[dict[str, Any]]:
    evidence_by_key = _unwrap_artifact_evidence(artifact_evidence)
    cases: list[dict[str, Any]] = []
    for contract in _select_contracts(plan, source_trace_ids):
        dataset_item_id = str(contract.get("dataset_item_id") or "")
        source_trace_id = str(contract.get("source_trace_id") or "")
        evidence = evidence_by_key.get(dataset_item_id) or evidence_by_key.get(source_trace_id) or {}
        summary = _evidence_summary(evidence)
        cases.append({
            "dataset_item_id": dataset_item_id,
            "source_trace_id": source_trace_id,
            "promotion_reason": str(contract.get("promotion_reason") or "unknown"),
            "manual_checks": _manual_check_names(contract),
            "evidence_summary": summary,
            "prompt": _build_prompt(contract, summary),
        })
    return cases


def build_scrubbed_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    for key in LANGFUSE_ENV_KEYS:
        env[key] = ""
    env["HERMES_SAVE_SESSION"] = "0"
    env["HERMES_LANGFUSE_SAMPLE_RATE"] = "0"
    env["HERMES_LANGFUSE_DEBUG"] = "false"
    return env


def build_hermes_command(prompt: str) -> list[str]:
    return [
        "hermes",
        "chat",
        "-q",
        prompt,
        "--quiet",
        "--max-turns",
        "2",
        "--source",
        "langfuse-local-replay",
        "--ignore-rules",
    ]


def subprocess_runner(command: Sequence[str], env: Mapping[str, str], timeout: int) -> ReplayRun:
    completed = subprocess.run(
        list(command),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return ReplayRun(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


RUNNER_STATUS_PREFIXES = (
    "Reached maximum iterations",
    "⚠️  Reached maximum iterations",
    "⚠️ Reached maximum iterations",
)


def _strip_runner_warning_prefix(line: str) -> str:
    stripped = line.strip()
    for prefix in ("⚠️  ", "⚠️ "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
    return stripped


def _is_runner_status_line(line: str) -> bool:
    normalized = _strip_runner_warning_prefix(line)
    return any(normalized.startswith(prefix) for prefix in RUNNER_STATUS_PREFIXES)


def _clean_stdout(stdout: str) -> CleanedCandidateOutput:
    candidate_lines: list[str] = []
    runner_status_messages: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if _is_runner_status_line(line):
            runner_status_messages.append(_strip_runner_warning_prefix(line))
            continue
        candidate_lines.append(line)
    # Quiet mode should already suppress banners/session metadata, but keep the
    # durable candidate concise and stable if the CLI adds blank lines or bounded
    # runner status warnings before the actual candidate text.
    return CleanedCandidateOutput(
        text="\n".join(candidate_lines).strip(),
        runner_status_messages=runner_status_messages,
    )


def run_replay_outputs(
    plan: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any],
    *,
    source_trace_ids: Sequence[str] = DEFAULT_SOURCE_TRACE_IDS,
    runner: Callable[[Sequence[str], Mapping[str, str], int], ReplayRun] = subprocess_runner,
    timeout: int = 180,
) -> dict[str, Any]:
    cases = build_replay_prompts(plan, artifact_evidence, source_trace_ids=source_trace_ids)
    if len(cases) > MAX_REPLAY_CASES:
        raise ReplayAdapterError(
            f"refusing to run {len(cases)} replay cases; max supported no-write batch is {MAX_REPLAY_CASES}"
        )
    env = build_scrubbed_env(os.environ)
    candidate_outputs: dict[str, str] = {}
    output_cases: list[dict[str, Any]] = []
    failures = 0
    for case in cases:
        command = build_hermes_command(case["prompt"])
        run = runner(command, env, timeout)
        cleaned = _clean_stdout(run.stdout)
        output = cleaned.text
        status = "generated" if run.exit_code == 0 and output else "failed"
        if status == "generated":
            candidate_outputs[case["dataset_item_id"]] = output
        else:
            failures += 1
        output_cases.append({
            "dataset_item_id": case["dataset_item_id"],
            "source_trace_id": case["source_trace_id"],
            "promotion_reason": case["promotion_reason"],
            "manual_checks": case["manual_checks"],
            "evidence_summary": case["evidence_summary"],
            "candidate_output_status": status,
            "exit_code": run.exit_code,
            "runner_status_messages": cleaned.runner_status_messages,
            "stderr_preview": run.stderr[:500] if run.stderr else "",
            "command_shape": ["hermes", "chat", "-q", "<prompt>", "--quiet", "--max-turns", "2", "--source", "langfuse-local-replay", "--ignore-rules"],
        })
    return {
        "mode": "local_replay_candidate_outputs_no_write",
        "write_enabled": False,
        "summary": {
            "requested_case_count": len(cases),
            "generated_output_count": len(candidate_outputs),
            "failed_case_count": failures,
        },
        "candidate_outputs": candidate_outputs,
        "cases": output_cases,
        "privacy": {
            "langfuse_env_scrubbed_for_child": True,
            "raw_trace_payloads_in_prompt": False,
            "raw_tool_payloads_in_prompt": False,
            "session_save_disabled_env": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--artifact-evidence-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--source-trace-id", action="append", dest="source_trace_ids")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)

    source_trace_ids = tuple(args.source_trace_ids or DEFAULT_SOURCE_TRACE_IDS)
    result = run_replay_outputs(
        _load_json(args.plan_json),
        _load_json(args.artifact_evidence_json),
        source_trace_ids=source_trace_ids,
        timeout=args.timeout,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"]["failed_case_count"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
