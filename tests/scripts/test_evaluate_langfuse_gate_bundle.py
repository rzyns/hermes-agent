"""Tests for conservative LF4 gate evaluation over sanitized bundles."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_langfuse_gate_bundle.py"
POLICY_PATH = Path("/home/openclaw/.hermes/artifacts/hermes-agent/langfuse-quality-next/lf4-30-conservative-gate-policy-schema-2026-05-07.json")


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_langfuse_gate_bundle", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sanitized_bundle(**overrides):
    bundle = {
        "manifest": {"run_name": "lf4-pilot-no-write", "run_spec_digest": "digest-123"},
        "privacy": {
            "privacy_safe": True,
            "raw_payloads_persisted": False,
            "secret_findings": 0,
            "proof": "artifact://secret-scan-summary",
        },
        "deterministic_checks": [
            {"id": "schema_validation", "required": True, "status": "passed", "proof": "artifact://schema-validation"},
            {"id": "secret_scan", "required": True, "status": "passed", "proof": "artifact://secret-scan-summary"},
        ],
        "manual_checks": [
            {"id": "semantic_review", "blocking": True, "status": "passed", "proof": "artifact://review-ledger"},
        ],
        "score_readback": {
            "required": False,
            "status": "not_applicable",
            "rationale": "no live score writes in this no-write run",
            "proof": "artifact://score-policy",
        },
        "idempotency": {"status": "passed", "proof": "artifact://run-spec-digest"},
        "approval_scope": {"status": "passed", "proof": "artifact://human-approval"},
        "side_effects": {"prohibited_side_effects": [], "proof": "artifact://no-write-manifest"},
        "observable_outcome": {"status": "passed", "proof": "artifact://local-eval-summary"},
    }
    bundle.update(overrides)
    return bundle


def blocker_ids(report):
    return {gate["id"] for gate in report["hard_gates"] if gate["blocks_success"]}


def score(report, name):
    return report["score_decisions"][name]


def test_evaluate_bundle_allows_success_only_when_privacy_and_blocking_checks_pass():
    script = load_script()

    report = script.evaluate_gate_bundle(sanitized_bundle(), script.load_policy(POLICY_PATH))

    assert report["mode"] == "langfuse_gate_evaluation_no_write"
    assert report["write_enabled"] is False
    assert report["suite_status"] == "passed"
    assert report["summary"] == {"hard_gate_count": 10, "blocking_gate_count": 0, "warning_count": 0}
    assert blocker_ids(report) == set()
    assert score(report, "privacy_safe")["value"] == 1
    assert score(report, "privacy_safe")["status"] == "passed"
    assert score(report, "privacy_safe")["proof"] == ["artifact://secret-scan-summary"]
    assert score(report, "test_passed")["value"] == 1
    assert score(report, "task_success")["value"] == 1
    assert "privacy_safe=1" in score(report, "task_success")["rationale"]


def test_task_success_never_passes_without_privacy_safe():
    script = load_script()
    unsafe = sanitized_bundle(privacy={"privacy_safe": False, "raw_payloads_persisted": False, "secret_findings": 1, "proof": "artifact://secret-scan-summary"})

    report = script.evaluate_gate_bundle(unsafe, script.load_policy(POLICY_PATH))

    assert "privacy_failure" in blocker_ids(report)
    assert report["suite_status"] == "failed"
    assert score(report, "privacy_safe")["value"] == 0
    assert score(report, "test_passed")["value"] == 0
    assert score(report, "task_success")["value"] == 0
    assert "privacy_safe=0" in score(report, "task_success")["rationale"]


def test_unclear_or_missing_blocking_manual_check_blocks_test_and_task_success():
    script = load_script()
    unclear = sanitized_bundle(manual_checks=[{"id": "semantic_review", "blocking": True, "status": "needs_manual_review", "proof": "artifact://review-ledger"}])
    missing = sanitized_bundle(manual_checks=[])

    unclear_report = script.evaluate_gate_bundle(unclear, script.load_policy(POLICY_PATH))
    missing_report = script.evaluate_gate_bundle(missing, script.load_policy(POLICY_PATH))

    assert "unresolved_manual_blocking_check" in blocker_ids(unclear_report)
    assert unclear_report["suite_status"] == "needs_manual_review"
    assert score(unclear_report, "test_passed")["value"] == 0
    assert score(unclear_report, "task_success")["value"] == 0
    assert "manual" in score(unclear_report, "task_success")["rationale"]
    assert "unresolved_manual_blocking_check" in blocker_ids(missing_report)
    assert missing_report["suite_status"] == "needs_manual_review"


def test_missing_required_deterministic_check_and_unknown_status_fail_closed():
    script = load_script()
    no_checks = sanitized_bundle(deterministic_checks=[])
    unknown = sanitized_bundle(deterministic_checks=[{"id": "schema_validation", "required": True, "status": "maybe", "proof": "artifact://schema-validation"}])

    no_checks_report = script.evaluate_gate_bundle(no_checks, script.load_policy(POLICY_PATH))
    unknown_report = script.evaluate_gate_bundle(unknown, script.load_policy(POLICY_PATH))

    assert "missing_required_deterministic_check" in blocker_ids(no_checks_report)
    assert no_checks_report["suite_status"] == "not_run"
    assert "missing_required_deterministic_check" in blocker_ids(unknown_report)
    assert unknown_report["suite_status"] == "insufficient_evidence"
    assert score(unknown_report, "test_passed")["value"] == 0


def test_cli_writes_machine_summary_markdown_and_claim_ledger(tmp_path, capsys):
    script = load_script()
    bundle_path = tmp_path / "bundle.json"
    output_json = tmp_path / "gate-results.json"
    output_md = tmp_path / "gate-summary.md"
    claim_ledger = tmp_path / "claim-ledger.json"
    bundle_path.write_text(json.dumps(sanitized_bundle()))

    exit_code = script.main([
        "--policy-json", str(POLICY_PATH),
        "--bundle-json", str(bundle_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
        "--claim-ledger-json", str(claim_ledger),
    ])

    assert exit_code == 0
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_json.read_text())
    ledger = json.loads(claim_ledger.read_text())
    assert stdout_report == file_report
    assert output_md.read_text().startswith("# LF4 conservative gate evaluation")
    assert file_report["claim_ledger_path"] == str(claim_ledger)
    assert {claim["claim"] for claim in ledger["claims"]} >= {"privacy_safe", "test_passed", "task_success"}
    assert all(claim["status"] == "supported" for claim in ledger["claims"])
    assert ledger["unsupported_claims"] == []
