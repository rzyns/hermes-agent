"""Tests for Kanban budget telemetry: complexity proxy and overhead envelope."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.observability import budget_telemetry as bt


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Complexity proxy
# ---------------------------------------------------------------------------


def test_estimate_task_complexity_minimum() -> None:
    proxy = bt.estimate_task_complexity(title="x", body="")
    assert proxy["score"] >= 1.0
    assert proxy["version"] == 1
    assert proxy["words"] == 0


def test_estimate_task_complexity_grows_with_content() -> None:
    body = "Implement feature.\n\n" + "word " * 300 + "\n```python\nprint(1)\n```\n"
    proxy = bt.estimate_task_complexity(
        title="Big feature",
        body=body,
        parents=["p1", "p2"],
        children=["c1"],
    )
    assert proxy["score"] > 1.0
    assert proxy["words"] >= 300
    assert proxy["lines"] >= 4
    assert proxy["code_blocks"] >= 1
    assert proxy["parents"] == 2
    assert proxy["children"] == 1


def test_complexity_proxy_round_trips_on_task(kanban_home: Path) -> None:
    conn = kb.connect()
    try:
        body = "Refactor the parser to support nested blocks.\n\n```python\nparse()\n```"
        tid = kb.create_task(conn, title="parser refactor", body=body, assignee="worker")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.metadata is not None
        proxy = task.metadata["complexity_proxy"]
        assert proxy["score"] > 1.0
        assert proxy["words"] > 0

        created = [e for e in kb.list_events(conn, tid) if e.kind == "created"]
        assert len(created) == 1
        assert created[0].payload is not None
        assert created[0].payload["complexity_proxy"]["score"] == proxy["score"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Overhead envelope
# ---------------------------------------------------------------------------


def _make_agent(**kwargs: Any) -> Any:
    defaults = {
        "max_iterations": 90,
        "iteration_budget": SimpleNamespace(used=87, max_total=90),
        "session_input_tokens": 1200,
        "session_output_tokens": 400,
        "session_total_tokens": 1600,
        "session_cache_read_tokens": 0,
        "session_cache_write_tokens": 0,
        "session_reasoning_tokens": 0,
        "context_compressor": SimpleNamespace(last_prompt_tokens=1100),
        "_last_turn_usage": {"total_tokens": 250},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_overhead_envelope_counts_messages() -> None:
    agent = _make_agent()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "t1"},
        {"role": "assistant", "content": "done"},
    ]
    env = bt.build_overhead_envelope(agent, api_call_count=2, messages=messages)
    assert env["api_call_count"] == 2
    assert env["max_iterations"] == 90
    assert env["budget_used"] == 87
    assert env["budget_max"] == 90
    assert env["tool_call_turns"] == 1
    assert env["assistant_messages"] == 2
    assert env["user_messages"] == 1
    assert env["message_count"] == 5
    assert env["input_tokens"] == 1200
    assert env["total_tokens"] == 1600
    assert env["last_prompt_tokens"] == 1100
    assert env["last_turn_usage"] == {"total_tokens": 250}


def test_build_overhead_envelope_tolerates_missing_attrs() -> None:
    agent = SimpleNamespace(max_iterations=50)
    env = bt.build_overhead_envelope(agent, api_call_count=1)
    assert env["api_call_count"] == 1
    assert env["max_iterations"] == 50
    assert env["budget_used"] == 0
    assert env["budget_max"] == 0
    assert env["last_prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# Integration: _record_task_failure enriches telemetry
# ---------------------------------------------------------------------------


def test_gave_up_event_includes_complexity_and_overhead(kanban_home: Path) -> None:
    conn = kb.connect()
    try:
        body = "Fix the bug.\n\n```python\nfoo()\n```"
        tid = kb.create_task(conn, title="bug fix", body=body, assignee="worker")
        kb.claim_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert run is not None

        overhead = {"api_call_count": 90, "max_iterations": 90, "budget_used": 90}
        tripped = kb._record_task_failure(
            conn,
            tid,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            expected_run_id=run.id,
            failure_limit=1,
            event_payload_extra={"overhead_envelope": overhead},
        )
        assert tripped

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "budget"

        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        payload = gave_up[0].payload
        assert payload is not None
        assert payload["complexity_proxy"]["score"] > 1.0
        assert payload["overhead_envelope"]["api_call_count"] == 90

        run2 = kb.latest_run(conn, tid)
        assert run2 is not None
        meta2 = run2.metadata
        assert meta2 is not None
        assert meta2["complexity_proxy"]["score"] > 1.0
        assert meta2["overhead_envelope"]["budget_used"] == 90
    finally:
        conn.close()
