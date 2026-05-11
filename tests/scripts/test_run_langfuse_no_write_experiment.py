"""Tests for Phase 4 no-write Langfuse experiment runner wrapper."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_langfuse_no_write_experiment.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_langfuse_no_write_experiment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def manifest(tmp_path: Path, *, case_count: int = 2, artifact_exists: bool = True) -> dict:
    candidate_artifact = tmp_path / "candidate-results.json"
    evidence_artifact = tmp_path / "semantic-adjudication.json"
    if artifact_exists:
        write_json(candidate_artifact, {
            "mode": "local_eval_no_write",
            "write_enabled": False,
            "summary": {"pass_count": 1, "needs_manual_review_count": 1},
            "items": [
                {"dataset_item_id": "item-1", "status": "pass", "candidate_output": "must not be copied"},
                {"dataset_item_id": "item-2", "status": "needs_manual_review", "candidate_text": "must not be copied either"},
            ],
        })
        write_json(evidence_artifact, {
            "mode": "semantic_adjudication_no_write",
            "write_enabled": False,
            "summary": {"approved": 1, "manual_pending": 1, "secret_findings": 0},
        })
    return {
        "schema": "hermes.langfuse.phase4.experiment_manifest.v1",
        "manifest_id": "test-manifest",
        "artifact_root": str(tmp_path / "runs" / "{run_spec_digest}"),
        "dataset": {
            "dataset_name": "hermes/turn-regression/pilot",
            "cases": [
                {"dataset_item_id": f"item-{idx}", "source_trace_id": f"trace-{idx}"}
                for idx in range(1, case_count + 1)
            ],
        },
        "idempotency": {"run_spec_digest": "abc123", "run_name": "test-run"},
        "inputs": {
            "candidate": {"candidate_artifacts": [{"path": str(candidate_artifact)}]},
            "evidence": {"evidence_artifacts": [{"path": str(evidence_artifact)}]},
        },
        "safety": {
            "langfuse_write_default": "blocked",
            "credential_value_persistence_allowed": False,
            "raw_trace_or_tool_payload_persistence_allowed": False,
        },
        "write_intent": {"mode": "no_write", "live_write_flags_allowed": False},
    }


def test_build_no_write_bundle_creates_redacted_evidence_without_live_writes(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-supersecret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-publicsecret")
    manifest_path = write_json(tmp_path / "manifest.json", manifest(tmp_path))
    bundle_dir = tmp_path / "bundle"

    report = script.build_no_write_bundle(manifest_path, bundle_dir, max_batch_size=30)
    serialized = json.dumps(report, sort_keys=True)
    runner_report = json.loads((bundle_dir / "runner-report.json").read_text())
    command_log = json.loads((bundle_dir / "command-log.json").read_text())

    assert report["mode"] == "phase4_no_write_experiment_runner"
    assert report["write_enabled"] is False
    assert report["langfuse_writes_attempted"] is False
    assert report["summary"]["case_count"] == 2
    assert report["summary"]["missing_artifact_count"] == 0
    assert report["artifacts"]["runner_report"].endswith("runner-report.json")
    assert runner_report["runner_status"] == "completed"
    assert runner_report["candidate_results"] == [
        {"dataset_item_id": "item-1", "status": "pass"},
        {"dataset_item_id": "item-2", "status": "needs_manual_review"},
    ]
    assert all("candidate_output" not in item and "candidate_text" not in item for item in runner_report["candidate_results"])
    assert command_log["steps"][0]["status"] == "completed"
    assert runner_report["summary"]["omitted_sensitive_named_summary_field_count"] == 1
    assert "secret_findings" not in json.dumps(runner_report, sort_keys=True)
    assert "***" not in serialized
    assert "pk-lf-publicsecret" not in serialized
    assert "must not be copied" not in json.dumps(runner_report, sort_keys=True)


def test_missing_artifact_fails_closed_before_bundle_completion(tmp_path):
    script = load_script()
    manifest_path = write_json(tmp_path / "manifest.json", manifest(tmp_path, artifact_exists=False))

    try:
        script.build_no_write_bundle(manifest_path, tmp_path / "bundle", max_batch_size=30)
    except script.NoWriteRunnerError as exc:
        assert "missing artifact" in str(exc)
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("missing artifacts must fail closed")


def test_oversized_batch_fails_closed(tmp_path):
    script = load_script()
    manifest_path = write_json(tmp_path / "manifest.json", manifest(tmp_path, case_count=31))

    try:
        script.build_no_write_bundle(manifest_path, tmp_path / "bundle", max_batch_size=30)
    except script.NoWriteRunnerError as exc:
        assert "oversized batch" in str(exc)
    else:  # pragma: no cover - documents fail-closed expectation
        raise AssertionError("oversized batches must fail closed")


def test_cli_writes_bundle_and_prints_status_separate_from_candidate_results(tmp_path, capsys, monkeypatch):
    script = load_script()
    monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-hidden")
    manifest_path = write_json(tmp_path / "manifest.json", manifest(tmp_path))
    bundle_dir = tmp_path / "bundle"

    exit_code = script.main([
        "--manifest-json", str(manifest_path),
        "--bundle-dir", str(bundle_dir),
    ])
    stdout_report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout_report["runner_status"] == "completed"
    assert stdout_report["write_enabled"] is False
    assert "candidate_results" not in stdout_report
    assert (bundle_dir / "runner-report.json").exists()
    assert "sk-lf-hidden" not in json.dumps(stdout_report, sort_keys=True)
