"""Tests for read-only Langfuse dataset evaluation planning."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "plan_langfuse_dataset_eval.py"


def load_script():
    spec = importlib.util.spec_from_file_location("plan_langfuse_dataset_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_item(**overrides):
    base = {
        "id": "hermes-item-1",
        "datasetName": "hermes/turn-regression/pilot",
        "sourceTraceId": "trace_1",
        "input": {
            "request": "Summarize the checked file without exposing secrets.",
            "context": {"platform": "discord", "workflow": "research"},
        },
        "expectedOutput": {
            "must": ["mention the reviewed evidence path"],
            "must_not": ["include API keys or bearer tokens"],
            "checks": ["privacy_safe", "grounded_summary"],
        },
        "metadata": {
            "promotion_reason": "privacy_case",
            "human_review": {"decision": "approved"},
        },
        "status": "ACTIVE",
    }
    base.update(overrides)
    return base


def test_build_plan_summarizes_dataset_items_without_running_experiments():
    script = load_script()

    plan = script.build_plan([dataset_item()], dataset_name="hermes/turn-regression/pilot")

    assert plan["mode"] == "read_only_eval_plan"
    assert plan["dataset_name"] == "hermes/turn-regression/pilot"
    assert plan["summary"] == {
        "dataset_item_count": 1,
        "active_item_count": 1,
        "items_with_source_trace_id": 1,
        "items_with_expected_checks": 1,
        "items_with_human_review": 1,
        "secret_findings": 0,
    }
    assert plan["proposed_experiment"] == {
        "write_enabled": False,
        "requires_explicit_future_flags": ["--write", "--confirm-experiment-write"],
        "scoring_policy": "deterministic_and_manual_first",
    }
    assert plan["contracts"] == [
        {
            "dataset_item_id": "hermes-item-1",
            "source_trace_id": "trace_1",
            "status": "ACTIVE",
            "promotion_reason": "privacy_case",
            "must": ["mention the reviewed evidence path"],
            "must_not": ["include API keys or bearer tokens"],
            "checks": ["privacy_safe", "grounded_summary"],
            "deterministic_checks": [
                {"name": "privacy_safe", "type": "secret_scan", "target": "candidate_output"},
                {"name": "grounded_summary", "type": "manual_review", "target": "candidate_output"},
            ],
        }
    ]


def test_cli_reads_local_dataset_item_payload_and_writes_plan(tmp_path, capsys):
    script = load_script()
    input_file = tmp_path / "dataset-items.json"
    output_file = tmp_path / "plan.json"
    input_file.write_text(json.dumps({"data": [dataset_item()]}))

    exit_code = script.main([
        "--input-json", str(input_file),
        "--dataset-name", "hermes/turn-regression/pilot",
        "--output-json", str(output_file),
    ])
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_file.read_text())

    assert exit_code == 0
    assert stdout_report == file_report
    assert file_report["summary"]["dataset_item_count"] == 1


def test_cli_live_mode_fetches_dataset_items_read_only_and_prints_credential_presence(monkeypatch, capsys):
    script = load_script()

    calls = []

    def fake_fetch(env, *, dataset_name, limit, page):
        calls.append({"dataset_name": dataset_name, "limit": limit, "page": page, "env": dict(env)})
        assert env["LANGFUSE_PUBLIC_KEY"] == "pk-lf-" + "publicsecret"
        assert env["LANGFUSE_SECRET_KEY"] == "sk-lf-" + "secretsecret"
        return {"data": [dataset_item()]}

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-" + "publicsecret")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-" + "secretsecret")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.local")
    monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-" + "publicsecret")
    monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-" + "secretsecret")
    monkeypatch.setenv("HERMES_LANGFUSE_BASE_URL", "https://langfuse.local")

    exit_code = script.main([
        "--dataset-name", "hermes/turn-regression/pilot",
        "--live-read",
        "--limit", "15",
    ], fetch=fake_fetch)
    output = capsys.readouterr().out
    report = json.loads(output)

    assert exit_code == 0
    assert calls[0]["dataset_name"] == "hermes/turn-regression/pilot"
    assert calls[0]["limit"] == 15
    assert calls[0]["page"] == 1
    assert report["credential_presence"] == {
        "public_key_present": True,
        "public_key_length": len("pk-lf-publicsecret"),
        "secret_key_present": True,
        "secret_key_length": len("sk-lf-secretsecret"),
        "host_present": True,
    }
    assert "pk-lf-publicsecret" not in output
    assert "sk-lf-secretsecret" not in output


def test_plan_flags_secret_like_contract_payload_without_leaking_value():
    script = load_script()
    item = dataset_item(input={"request": "use token=supersecret", "context": {}})

    plan = script.build_plan([item], dataset_name="hermes/turn-regression/pilot")
    serialized = json.dumps(plan, sort_keys=True)

    assert plan["summary"]["secret_findings"] == 1
    assert "supersecret" not in serialized
    assert plan["secret_findings"] == [
        {
            "dataset_item_id": "hermes-item-1",
            "path": "item.input.request",
            "pattern": "generic_assignment_secret",
            "preview": "use token[REDACTED]",
        }
    ]
