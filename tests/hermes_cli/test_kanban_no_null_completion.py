"""Tests for the no-null completion guard (P3.3e).

Covers both the DB-layer helpers (complete_task / edit_completed_task_result)
and the CLI entry points that surface them to users.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# complete_task effective-result guard
# ---------------------------------------------------------------------------

def test_complete_task_blocks_when_both_result_and_summary_are_none(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        with pytest.raises(kb.CompletionResultRequiredError, match=str(t)):
            kb.complete_task(conn, t, result=None, summary=None)
        task = kb.get_task(conn, t)
    assert task is not None
    assert task.status != "done"


def test_complete_task_blocks_when_both_result_and_summary_are_whitespace(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        with pytest.raises(kb.CompletionResultRequiredError, match=str(t)):
            kb.complete_task(conn, t, result="", summary="  ")
        task = kb.get_task(conn, t)
    assert task is not None
    assert task.status != "done"


def test_complete_task_succeeds_when_only_summary_is_substantive(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        assert kb.complete_task(conn, t, result=None, summary="substantive")
        task = kb.get_task(conn, t)
    assert task is not None
    assert task.status == "done"
    assert task.result == "substantive"


def test_complete_task_succeeds_when_only_result_is_substantive(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        assert kb.complete_task(conn, t, result="substantive", summary=None)
        task = kb.get_task(conn, t)
    assert task is not None
    assert task.status == "done"
    assert task.result == "substantive"


# ---------------------------------------------------------------------------
# edit_completed_task_result guard
# ---------------------------------------------------------------------------

def test_edit_completed_task_result_blocks_blank_result(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.complete_task(conn, t, result="original")
        with pytest.raises(kb.CompletionResultRequiredError, match=str(t)):
            kb.edit_completed_task_result(conn, t, result="")
        task = kb.get_task(conn, t)
    assert task is not None
    assert task.result == "original"
    assert task.status == "done"


# ---------------------------------------------------------------------------
# CLI surface guard
# ---------------------------------------------------------------------------

def test_cli_complete_without_result_or_summary_exits_nonzero(kanban_home):
    out = kc.run_slash("create 'x'")
    import re
    m = re.search(r"(t_[a-f0-9]+)", out)
    assert m, f"no task id in: {out!r}"
    tid = m.group(1)

    rc = kc._cmd_complete(
        argparse.Namespace(task_ids=[tid], result=None, summary=None, metadata=None)
    )
    assert rc == 2


def test_run_slash_complete_without_result_or_summary_returns_error(kanban_home):
    out = kc.run_slash("create 'x'")
    import re
    m = re.search(r"(t_[a-f0-9]+)", out)
    assert m, f"no task id in: {out!r}"
    tid = m.group(1)

    result = kc.run_slash(f"complete {tid}")
    assert "completion blocked" in result.lower()
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status != "done"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
