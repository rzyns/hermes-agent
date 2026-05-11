#!/usr/bin/env python3
"""Evaluate conservative LF4 gates over sanitized run/evidence bundles.

This helper is intentionally no-write. It consumes sanitized manifests and compact
evidence/status ledgers only; raw trace/tool payloads are not required and are not
copied into outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


STATUSES = {"passed", "failed", "needs_manual_review", "not_run", "not_applicable", "insufficient_evidence"}
SCORE_NAMES = ("privacy_safe", "test_passed", "task_success")


class GateEvaluationError(ValueError):
    """Raised when gate evaluation inputs are invalid."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path)
    if not isinstance(policy, dict):
        raise GateEvaluationError("policy JSON must be an object")
    blockers = policy.get("hard_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise GateEvaluationError("policy JSON is missing hard_blockers")
    return policy


def _proof(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item]
    return []


def _status(value: Any) -> str:
    status = str(value or "").strip()
    if status in STATUSES:
        return status
    return "insufficient_evidence"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _hard_blocker_policy(policy: Mapping[str, Any], blocker_id: str) -> Mapping[str, Any]:
    for blocker in policy.get("hard_blockers", []):
        if isinstance(blocker, Mapping) and blocker.get("id") == blocker_id:
            return blocker
    return {"id": blocker_id, "result_status": "failed", "rationale": "policy entry not found"}


def _gate(
    policy: Mapping[str, Any],
    blocker_id: str,
    status: str,
    *,
    proof: Sequence[str] | None = None,
    detail: str = "",
) -> dict[str, Any]:
    blocker = _hard_blocker_policy(policy, blocker_id)
    expected_block_status = _status(blocker.get("result_status") or "failed")
    blocks_success = status == expected_block_status or status in {"failed", "needs_manual_review", "not_run", "insufficient_evidence"}
    if status in {"passed", "not_applicable"}:
        blocks_success = False
    return {
        "id": blocker_id,
        "status": status,
        "blocks_success": blocks_success,
        "blocks_scores": list(blocker.get("blocks_scores", [])) if isinstance(blocker.get("blocks_scores"), list) else [],
        "rationale": str(blocker.get("rationale") or ""),
        "detail": detail,
        "proof": list(proof or []),
    }


def _evaluate_privacy(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    privacy = _mapping(bundle.get("privacy"))
    secret_findings = int(privacy.get("secret_findings", 0) or 0)
    raw_payloads = bool(privacy.get("raw_payloads_persisted"))
    privacy_safe = privacy.get("privacy_safe") is True and secret_findings == 0 and not raw_payloads
    proof = _proof(privacy.get("proof"))
    return [
        _gate(
            policy,
            "privacy_failure",
            "passed" if privacy_safe else "failed",
            proof=proof,
            detail="privacy evidence is safe" if privacy_safe else "privacy_safe evidence is false, missing, or secret findings are present",
        ),
        _gate(
            policy,
            "raw_payload_leak",
            "failed" if raw_payloads else "passed",
            proof=proof,
            detail="raw payloads persisted" if raw_payloads else "no raw payload persistence indicated",
        ),
    ]


def _evaluate_deterministic(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = _list_of_mappings(bundle.get("deterministic_checks"))
    required = [check for check in checks if check.get("required", True) is not False]
    proof = [item for check in required for item in _proof(check.get("proof"))]
    if not required:
        return [
            _gate(policy, "failed_deterministic_check", "passed", proof=[], detail="no failed deterministic check evidence"),
            _gate(policy, "missing_required_deterministic_check", "not_run", proof=[], detail="no required deterministic checks were provided"),
        ]
    statuses = [_status(check.get("status")) for check in required]
    has_failed = any(status == "failed" for status in statuses)
    has_missing_or_unknown = any(status in {"not_run", "insufficient_evidence"} for status in statuses)
    return [
        _gate(
            policy,
            "failed_deterministic_check",
            "failed" if has_failed else "passed",
            proof=proof,
            detail="at least one required deterministic check failed" if has_failed else "no required deterministic check failed",
        ),
        _gate(
            policy,
            "missing_required_deterministic_check",
            "insufficient_evidence" if has_missing_or_unknown else "passed",
            proof=proof,
            detail="a required deterministic check is not_run, missing, or unknown" if has_missing_or_unknown else "all required deterministic checks have terminal passing evidence",
        ),
    ]


def _evaluate_score_readback(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    readback = _mapping(bundle.get("score_readback"))
    required = bool(readback.get("required"))
    status = _status(readback.get("status"))
    proof = _proof(readback.get("proof"))
    if not required:
        missing_status = "not_applicable" if status == "not_applicable" and readback.get("rationale") and proof else "passed"
        mismatch_status = "passed"
    else:
        missing_status = "passed" if status == "passed" and proof else "not_run"
        mismatch_status = "failed" if status == "failed" else "passed"
    return [
        _gate(policy, "missing_required_score_or_readback", missing_status, proof=proof, detail=str(readback.get("rationale") or "score readback requirement evaluated")),
        _gate(policy, "score_readback_mismatch", mismatch_status, proof=proof, detail="score readback mismatch" if mismatch_status == "failed" else "no score readback mismatch indicated"),
    ]


def _evaluate_manual(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    checks = _list_of_mappings(bundle.get("manual_checks"))
    blocking = [check for check in checks if check.get("blocking", True) is not False]
    proof = [item for check in blocking for item in _proof(check.get("proof"))]
    if not blocking:
        status = "needs_manual_review"
        detail = "no blocking manual check evidence was provided"
    else:
        statuses = [_status(check.get("status")) for check in blocking]
        unresolved = any(status != "passed" for status in statuses)
        status = "needs_manual_review" if unresolved else "passed"
        detail = "a blocking manual check is missing, unclear, pending, or needs review" if unresolved else "all blocking manual checks passed"
    return _gate(policy, "unresolved_manual_blocking_check", status, proof=proof, detail=detail)


def _single_status_gate(bundle: Mapping[str, Any], policy: Mapping[str, Any], bundle_key: str, blocker_id: str, failed_detail: str) -> dict[str, Any]:
    section = _mapping(bundle.get(bundle_key))
    status = _status(section.get("status"))
    proof = _proof(section.get("proof"))
    gate_status = "passed" if status == "passed" and proof else ("failed" if status == "failed" else "insufficient_evidence")
    return _gate(policy, blocker_id, gate_status, proof=proof, detail="passed" if gate_status == "passed" else failed_detail)


def _evaluate_side_effects(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(bundle.get("side_effects"))
    effects = section.get("prohibited_side_effects")
    has_effects = bool(effects) if isinstance(effects, list) else bool(effects)
    return _gate(
        policy,
        "prohibited_side_effect",
        "failed" if has_effects else "passed",
        proof=_proof(section.get("proof")),
        detail="prohibited side effects present" if has_effects else "no prohibited side effects indicated",
    )


def _suite_status(gates: Sequence[Mapping[str, Any]]) -> str:
    blocking = [gate for gate in gates if gate.get("blocks_success")]
    if not blocking:
        return "passed"
    statuses = {str(gate.get("status")) for gate in blocking}
    for status in ("failed", "needs_manual_review", "insufficient_evidence", "not_run"):
        if status in statuses:
            return status
    return "insufficient_evidence"


def _score_decisions(bundle: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    blockers_by_score: dict[str, list[Mapping[str, Any]]] = {name: [] for name in SCORE_NAMES}
    for gate in gates:
        if not gate.get("blocks_success"):
            continue
        for score_name in gate.get("blocks_scores", []):
            if score_name in blockers_by_score:
                blockers_by_score[score_name].append(gate)
    privacy_blocked = bool(blockers_by_score["privacy_safe"])
    test_blocked = privacy_blocked or bool(blockers_by_score["test_passed"])
    task_blocked = privacy_blocked or test_blocked or bool(blockers_by_score["task_success"])
    outcome = _mapping(bundle.get("observable_outcome"))
    outcome_status = _status(outcome.get("status"))
    outcome_proof = _proof(outcome.get("proof"))
    if outcome_status != "passed" or not outcome_proof:
        task_blocked = True
    def gate_proofs(score_name: str) -> list[str]:
        proofs: list[str] = []
        for gate in blockers_by_score.get(score_name, []):
            proofs.extend(str(item) for item in gate.get("proof", []) if item)
        return proofs
    privacy = _mapping(bundle.get("privacy"))
    privacy_proof = _proof(privacy.get("proof"))
    deterministic_proofs = [str(item) for gate in gates if gate.get("id") in {"failed_deterministic_check", "missing_required_deterministic_check"} for item in gate.get("proof", [])]
    manual_proofs = [str(item) for gate in gates if gate.get("id") == "unresolved_manual_blocking_check" for item in gate.get("proof", [])]
    def unique(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result
    return {
        "privacy_safe": {
            "value": 0 if privacy_blocked else 1,
            "status": "failed" if privacy_blocked else "passed",
            "proof": unique(gate_proofs("privacy_safe") or privacy_proof),
            "rationale": "privacy_safe=0 due to privacy/raw-payload blockers" if privacy_blocked else "privacy_safe=1: no privacy failure or raw-payload leak evidence",
        },
        "test_passed": {
            "value": 0 if test_blocked else 1,
            "status": "failed" if test_blocked else "passed",
            "proof": unique(gate_proofs("test_passed") or deterministic_proofs + manual_proofs),
            "rationale": "blocked by privacy_safe=0 or unresolved deterministic/manual hard blockers" if test_blocked else "required deterministic and blocking manual checks passed with privacy_safe=1",
        },
        "task_success": {
            "value": 0 if task_blocked else 1,
            "status": "failed" if task_blocked else "passed",
            "proof": unique(gate_proofs("task_success") or privacy_proof + deterministic_proofs + manual_proofs + outcome_proof),
            "rationale": "privacy_safe=0, test_passed=0, manual review unresolved, side effect present, or observable outcome missing" if task_blocked else "task_success=1 requires privacy_safe=1, test_passed=1, resolved blocking manual checks, no side effects, and observable outcome proof",
        },
    }


def evaluate_gate_bundle(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        raise GateEvaluationError("bundle JSON must be an object")
    gates: list[dict[str, Any]] = []
    gates.extend(_evaluate_privacy(bundle, policy))
    gates.extend(_evaluate_deterministic(bundle, policy))
    gates.extend(_evaluate_score_readback(bundle, policy))
    gates.append(_evaluate_manual(bundle, policy))
    gates.append(_single_status_gate(bundle, policy, "idempotency", "idempotency_mismatch", "idempotency evidence is missing or mismatched"))
    gates.append(_single_status_gate(bundle, policy, "approval_scope", "approval_scope_mismatch", "approval scope evidence is missing or mismatched"))
    gates.append(_evaluate_side_effects(bundle, policy))
    warnings: list[dict[str, Any]] = []
    score_decisions = _score_decisions(bundle, gates)
    suite_status = _suite_status(gates)
    return {
        "mode": "langfuse_gate_evaluation_no_write",
        "write_enabled": False,
        "schema": "hermes.langfuse.phase4.gate_evaluation.v1",
        "suite_status": suite_status,
        "summary": {
            "hard_gate_count": len(gates),
            "blocking_gate_count": sum(1 for gate in gates if gate.get("blocks_success")),
            "warning_count": len(warnings),
        },
        "hard_gates": gates,
        "warnings": warnings,
        "score_decisions": score_decisions,
    }


def build_claim_ledger(report: Mapping[str, Any]) -> dict[str, Any]:
    claims = []
    scores = _mapping(report.get("score_decisions"))
    for name in SCORE_NAMES:
        decision = _mapping(scores.get(name))
        claims.append({
            "claim": name,
            "status": "supported" if decision.get("proof") else "unsupported",
            "decision_status": decision.get("status"),
            "value": decision.get("value"),
            "evidence": [{"type": "proof_pointer", "ref": ref} for ref in decision.get("proof", [])],
        })
    claims.append({
        "claim": "suite_status",
        "status": "supported",
        "decision_status": report.get("suite_status"),
        "evidence": [{"type": "hard_gate_results", "count": report.get("summary", {}).get("hard_gate_count")}],
    })
    unsupported = [claim for claim in claims if claim.get("status") != "supported"]
    return {"schema": "hermes.langfuse.phase4.gate_claim_ledger.v1", "claims": claims, "unsupported_claims": unsupported}


def render_markdown(report: Mapping[str, Any], claim_ledger_path: Path | None = None) -> str:
    lines = [
        "# LF4 conservative gate evaluation",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- Write enabled: `{report.get('write_enabled')}`",
        f"- Suite status: `{report.get('suite_status')}`",
        f"- Hard gates: `{report.get('summary', {}).get('hard_gate_count')}` total, `{report.get('summary', {}).get('blocking_gate_count')}` blocking",
    ]
    if claim_ledger_path:
        lines.append(f"- Claim ledger: `{claim_ledger_path}`")
    lines.extend(["", "## Score decisions", ""])
    for name, decision in _mapping(report.get("score_decisions")).items():
        lines.append(f"- `{name}`: value `{decision.get('value')}`, status `{decision.get('status')}` — {decision.get('rationale')}")
    lines.extend(["", "## Blocking hard gates", ""])
    blocking = [gate for gate in report.get("hard_gates", []) if isinstance(gate, Mapping) and gate.get("blocks_success")]
    if not blocking:
        lines.append("- None")
    else:
        for gate in blocking:
            lines.append(f"- `{gate.get('id')}`: `{gate.get('status')}` — {gate.get('detail')}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate conservative LF4 gates over a sanitized evidence bundle.")
    parser.add_argument("--policy-json", type=Path, required=True, help="LF4-30 conservative gate policy JSON")
    parser.add_argument("--bundle-json", type=Path, required=True, help="Sanitized run/evidence bundle JSON")
    parser.add_argument("--output-json", type=Path, help="Write machine-readable gate result JSON")
    parser.add_argument("--output-md", type=Path, help="Write reviewer-readable Markdown summary")
    parser.add_argument("--claim-ledger-json", type=Path, help="Write claim ledger JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy_json)
        bundle = _load_json(args.bundle_json)
        report = evaluate_gate_bundle(bundle, policy)
        if args.claim_ledger_json:
            ledger = build_claim_ledger(report)
            args.claim_ledger_json.parent.mkdir(parents=True, exist_ok=True)
            args.claim_ledger_json.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
            report["claim_ledger_path"] = str(args.claim_ledger_json)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(report, args.claim_ledger_json))
        print(rendered)
        return 0
    except GateEvaluationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
