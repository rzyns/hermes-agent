"""Tests for tiny no-write Langfuse replay/model-output adapter."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_langfuse_replay_candidate_outputs.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_langfuse_replay_candidate_outputs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_plan():
    return {
        "dataset_name": "hermes/turn-regression/pilot",
        "contracts": [
            {
                "dataset_item_id": "item-success",
                "source_trace_id": "trace-success",
                "promotion_reason": "canonical_success",
                "must": ["this contract text must not be copied"],
                "must_not": ["this forbidden text must not be copied"],
                "deterministic_checks": [
                    {"name": "canonical_improved_trace_classified", "type": "manual_review"},
                    {"name": "tool_outputs_present", "type": "deterministic_artifact_check"},
                ],
            },
            {
                "dataset_item_id": "item-failure",
                "source_trace_id": "trace-failure",
                "promotion_reason": "failure",
                "must": ["another copied bullet"],
                "must_not": [],
                "deterministic_checks": [
                    {"name": "local_default_cli_trace_classified", "type": "manual_review"},
                    {"name": "all_sampled_tool_outputs_null", "type": "manual_review"},
                ],
            },
        ],
    }


def sample_evidence():
    return {
        "artifact_evidence": {
            "trace-success": {
                "summary": {
                    "trace_id": "trace-success",
                    "tool_observations": 47,
                    "tool_null_output_count": 0,
                    "tool_output_present_count": 47,
                    "tool_call_id_present_count": 47,
                    "tool_args_present_count": 47,
                },
                "tool_null_outputs_zero": True,
            },
            "trace-failure": {
                "summary": {
                    "trace_id": "trace-failure",
                    "tool_observations": 6,
                    "tool_null_output_count": 6,
                    "tool_output_present_count": 0,
                    "tool_call_id_present_count": 0,
                    "tool_args_present_count": 6,
                },
                "tool_null_outputs_zero": False,
            },
        }
    }


def test_build_replay_prompts_use_minimized_evidence_without_copying_contract_bullets():
    script = load_script()

    prompts = script.build_replay_prompts(sample_plan(), sample_evidence(), source_trace_ids=["trace-success", "trace-failure"])

    assert [case["source_trace_id"] for case in prompts] == ["trace-success", "trace-failure"]
    first_prompt = prompts[0]["prompt"]
    assert "tool_observations" in first_prompt
    assert "canonical_improved_trace_classified" in first_prompt
    assert "this contract text must not be copied" not in first_prompt
    assert "this forbidden text must not be copied" not in first_prompt
    assert "raw trace" in first_prompt.lower()
    assert prompts[0]["evidence_summary"]["tool_observations"] == 47


def test_scrubbed_replay_environment_removes_langfuse_credentials_and_session_saving(monkeypatch):
    script = load_script()
    monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "fake-public-key")
    monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "fake-secret-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")
    monkeypatch.setenv("OPENROUTER_API_KEY", "keep-model-key")

    env = script.build_scrubbed_env(os.environ)

    assert env["HERMES_LANGFUSE_PUBLIC_KEY"] == ""
    assert env["HERMES_LANGFUSE_SECRET_KEY"] == ""
    assert env["LANGFUSE_PUBLIC_KEY"] == ""
    assert env["LANGFUSE_SECRET_KEY"] == ""
    assert env["LANGFUSE_BASE_URL"] == ""
    assert env["HERMES_SAVE_SESSION"] == "0"
    assert env["OPENROUTER_API_KEY"] == "keep-model-key"


def test_build_hermes_command_is_quiet_bounded_and_source_tagged():
    script = load_script()

    command = script.build_hermes_command("Review this evidence")

    assert command[:3] == ["hermes", "chat", "-q"]
    assert "Review this evidence" in command
    assert "--quiet" in command
    assert command[command.index("--max-turns") + 1] == "2"
    assert command[command.index("--source") + 1] == "langfuse-local-replay"
    assert "--ignore-rules" in command


def test_run_replay_outputs_with_injected_runner_writes_envelope_without_langfuse_writes():
    script = load_script()
    calls = []

    def fake_runner(command, env, timeout):
        calls.append({"command": command, "env": env, "timeout": timeout})
        return script.ReplayRun(exit_code=0, stdout="Verdict: needs human review. Evidence supports success.", stderr="")

    result = script.run_replay_outputs(
        sample_plan(),
        sample_evidence(),
        source_trace_ids=["trace-success"],
        runner=fake_runner,
    )

    assert result["mode"] == "local_replay_candidate_outputs_no_write"
    assert result["write_enabled"] is False
    assert result["summary"] == {"requested_case_count": 1, "generated_output_count": 1, "failed_case_count": 0}
    assert result["candidate_outputs"] == {"item-success": "Verdict: needs human review. Evidence supports success."}
    assert result["cases"][0]["candidate_output_status"] == "generated"
    assert calls[0]["env"]["HERMES_LANGFUSE_SECRET_KEY"] == ""


def test_run_replay_outputs_separates_runner_status_noise_from_candidate_text():
    script = load_script()

    def noisy_runner(command, env, timeout):
        return script.ReplayRun(
            exit_code=0,
            stdout=(
                "⚠️  Reached maximum iterations (2). Requesting summary...\n"
                "Verdict: needs human review.\n"
                "Evidence: minimized artifact supports structural review only.\n"
            ),
            stderr="session_id: abc123\nstty: 'standard input': Inappropriate ioctl for device\n",
        )

    result = script.run_replay_outputs(
        sample_plan(),
        sample_evidence(),
        source_trace_ids=["trace-success"],
        runner=noisy_runner,
    )

    candidate = result["candidate_outputs"]["item-success"]
    case = result["cases"][0]
    assert candidate.startswith("Verdict: needs human review.")
    assert "Reached maximum iterations" not in candidate
    assert "session_id" not in candidate
    assert case["runner_status_messages"] == ["Reached maximum iterations (2). Requesting summary..."]
    assert "session_id: abc123" in case["stderr_preview"]


def test_run_replay_outputs_refuses_batches_above_guardrail():
    script = load_script()
    plan = sample_plan()
    plan["contracts"] = [
        {
            "dataset_item_id": f"item-{index}",
            "source_trace_id": f"trace-{index}",
            "promotion_reason": "failure",
            "deterministic_checks": [],
        }
        for index in range(31)
    ]

    try:
        script.run_replay_outputs(
            plan,
            {"artifact_evidence": {}},
            source_trace_ids=[f"trace-{index}" for index in range(31)],
        )
    except script.ReplayAdapterError as exc:
        assert "refusing to run 31 replay cases" in str(exc)
    else:  # pragma: no cover - clearer than pytest.raises for script-loaded module
        raise AssertionError("expected replay batch guardrail to reject oversized batch")
