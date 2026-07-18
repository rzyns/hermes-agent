"""Tests for the Kanban decomposer dry-run path.

All tests use mocked auxiliary client and profile stubs — no network,
no production DB writes.
"""

from __future__ import annotations

import json as jsonlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp
from hermes_cli.kanban import _cmd_decompose


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    return patch(
        "agent.auxiliary_client.get_auxiliary_extra_body",
        return_value={},
    )


def _patch_list_profiles(names: list[str]):
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n,
            is_default=(i == 0),
            description=f"desc for {n}",
            description_auto=False,
            model="m",
            provider="p",
            skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value=names[0] if names else "default",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Mutation gating — no DB writes in dry-run mode
# ---------------------------------------------------------------------------

class TestDryRunNoMutation:
    """Dry-run must not create tasks, flip status, or write comments/events."""

    def test_dry_run_fanout_leaves_db_unchanged(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="ship a feature", body="do it", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "test split",
            "tasks": [
                {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
                {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
            ],
        })

        patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
        for p in patches:
            p.start()
        try:
            with _patch_aux_client(llm_payload), _patch_extra_body():
                outcome = decomp.decompose_task(tid, author="me", dry_run=True)
        finally:
            for p in patches:
                p.stop()

        assert outcome.ok
        assert outcome.dry_run is True
        assert outcome.fanout is True
        assert outcome.child_ids is None
        assert len(outcome.dry_run_tasks) == 2

        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
            assert task.status == "triage"
            assert task.title == "ship a feature"
            links = conn.execute(
                "SELECT child_id FROM task_links WHERE parent_id = ?", (tid,)
            ).fetchall()
            assert len(links) == 0
            comments = kb.list_comments(conn, tid)
            assert len(comments) == 0
            events = kb.list_events(conn, tid)
            assert not any(ev.kind == "decomposed" for ev in events)
            assert not any(ev.kind == "promoted" for ev in events)
            assert not any("Decomposed into" in (ev.payload or {}) for ev in events)

    def test_dry_run_no_fanout_leaves_db_unchanged(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="just one thing", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": False,
            "rationale": "single unit",
            "title": "Tightened title",
            "body": "**Goal**\nDo the thing.",
            "assignee": "engineer",
        })

        patches = _patch_list_profiles(["orchestrator", "engineer"])
        for p in patches:
            p.start()
        try:
            with _patch_aux_client(llm_payload), _patch_extra_body():
                outcome = decomp.decompose_task(tid, author="me", dry_run=True)
        finally:
            for p in patches:
                p.stop()

        assert outcome.ok
        assert outcome.dry_run is True
        assert outcome.fanout is False
        assert outcome.new_title == "Tightened title"
        assert outcome.dry_run_tasks == [{
            "title": "Tightened title",
            "body": "**Goal**\nDo the thing.",
            "assignee": "engineer",
            "assignee_update": "engineer",
            "effective_assignee": "engineer",
        }]

        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
            assert task.status == "triage"
            assert task.title == "just one thing"
            assert not task.assignee
            comments = kb.list_comments(conn, tid)
            assert len(comments) == 0
            events = kb.list_events(conn, tid)
            assert not any(ev.kind == "decomposed" for ev in events)
            assert not any(ev.kind == "promoted" for ev in events)


    def test_dry_run_no_fanout_existing_assignee_reports_update_and_effective_assignee(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(
                conn,
                title="already routed",
                assignee="reviewer",
                triage=True,
            )

        llm_payload = jsonlib.dumps({
            "fanout": False,
            "rationale": "single unit",
            "title": "Tightened title",
            "body": "Do the thing.",
            "assignee": "engineer",
        })

        patches = _patch_list_profiles(["orchestrator", "reviewer", "engineer"])
        for p in patches:
            p.start()
        try:
            with _patch_aux_client(llm_payload), _patch_extra_body():
                outcome = decomp.decompose_task(tid, author="me", dry_run=True)
        finally:
            for p in patches:
                p.stop()

        assert outcome.ok
        assert outcome.dry_run is True
        assert outcome.dry_run_tasks == [{
            "title": "Tightened title",
            "body": "Do the thing.",
            "assignee": None,
            "assignee_update": None,
            "effective_assignee": "reviewer",
        }]

        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.status == "triage"
            assert task.title == "already routed"
            assert task.assignee == "reviewer"
            assert not kb.list_comments(conn, tid)
            events = kb.list_events(conn, tid)
            assert not any(ev.kind == "decomposed" for ev in events)
            assert not any(ev.kind == "promoted" for ev in events)


# ---------------------------------------------------------------------------
# 2. Prompt parity — real and dry-run share the same prompt-building path
# ---------------------------------------------------------------------------

class TestDryRunPromptParity:
    """The prompt, roster, and aux-client call must be identical up to the
    mutation gate so dry-run is a faithful preview of real decomposition."""

    def test_dry_run_and_real_prompt_identical(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="parity test", body="body text", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": False,
            "rationale": "single",
            "title": "Tightened",
            "body": "Do it.",
            "assignee": "engineer",
        })

        calls = []

        def capturing_call_llm(*, messages, **kwargs):
            calls.append(messages)
            return _fake_aux_response(llm_payload)

        patches = _patch_list_profiles(["orchestrator", "engineer"])
        for p in patches:
            p.start()
        try:
            with patch(
                "agent.auxiliary_client.call_llm",
                side_effect=capturing_call_llm,
            ), _patch_extra_body(), patch(
                "hermes_cli.kanban_decompose._load_config",
                return_value={},
            ):
                decomp.decompose_task(tid, author="me", dry_run=True)
                # For the "real" call, intercept the DB write so we don't mutate
                with patch(
                    "hermes_cli.kanban_decompose.kb.specify_triage_task",
                    return_value=True,
                ):
                    decomp.decompose_task(tid, author="me", dry_run=False)
        finally:
            for p in patches:
                p.stop()

        assert len(calls) == 2
        assert calls[0] == calls[1]


# ---------------------------------------------------------------------------
# 3. JSON output contract — parseable, marked, carries plan evidence
# ---------------------------------------------------------------------------

class TestDryRunJsonOutput:
    """``--json --dry-run`` must emit deterministic JSON with dry_run marker
    and enough child-graph detail for evidence packets."""

    def test_json_dry_run_fanout(self, kanban_home, capsys):
        import argparse

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="json test", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "json test",
            "tasks": [
                {"title": "A", "body": "a", "assignee": "engineer", "parents": []},
            ],
        })

        patches = _patch_list_profiles(["orchestrator", "engineer"])
        for p in patches:
            p.start()
        try:
            with _patch_aux_client(llm_payload), _patch_extra_body():
                args = argparse.Namespace(
                    task_id=tid,
                    all_triage=False,
                    tenant=None,
                    author="me",
                    json=True,
                    dry_run=True,
                )
                rc = _cmd_decompose(args)
        finally:
            for p in patches:
                p.stop()

        assert rc == 0
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().split("\n") if ln.strip()]
        assert len(lines) == 1
        payload = jsonlib.loads(lines[0])
        assert payload["ok"] is True
        assert payload["dry_run"] is True
        assert payload["fanout"] is True
        assert "dry_run_tasks" in payload
        assert len(payload["dry_run_tasks"]) == 1
        assert payload["dry_run_tasks"][0]["assignee"] == "engineer"
        assert "dry_run_parsed" in payload
        assert payload["dry_run_parsed"]["rationale"] == "json test"

    def test_json_dry_run_no_secrets_in_output(self, kanban_home, capsys):
        import argparse

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="secret test", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "secret test",
            "tasks": [
                {"title": "A", "body": "a", "assignee": "engineer", "parents": []},
            ],
        })

        patches = _patch_list_profiles(["orchestrator", "engineer"])
        for p in patches:
            p.start()
        try:
            with _patch_aux_client(llm_payload), _patch_extra_body():
                args = argparse.Namespace(
                    task_id=tid,
                    all_triage=False,
                    tenant=None,
                    author="me",
                    json=True,
                    dry_run=True,
                )
                rc = _cmd_decompose(args)
        finally:
            for p in patches:
                p.stop()

        assert rc == 0
        raw = capsys.readouterr().out
        # Structural check: must be valid JSON
        payload = jsonlib.loads(raw.strip().split("\n")[0])
        # Secret-safety: no obvious key-like strings in the serialized output
        raw_lower = raw.lower()
        for pat in ("sk-", "bearer ", "api_key", "token=", "secret="):
            assert pat not in raw_lower, f"possible secret leak: {pat!r} in output"
        # Ensure parsed payload itself doesn't carry the mock client object
        assert isinstance(payload.get("dry_run_parsed"), dict)
        assert isinstance(payload.get("dry_run_tasks"), list)


# ---------------------------------------------------------------------------
# 4. Real path preservation — non-dry-run still mutates as before
# ---------------------------------------------------------------------------

class TestRealPathUnchanged:
    """When ``dry_run=False`` (default), decomposition must continue to
    create children, flip the root to ``todo``, and record audit metadata."""

    def test_real_decompose_still_creates_children(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="real test", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "real",
            "tasks": [
                {"title": "child", "body": "c", "assignee": "engineer", "parents": []},
            ],
        })

        patches = _patch_list_profiles(["orchestrator", "engineer"])
        for p in patches:
            p.start()
        try:
            with _patch_aux_client(llm_payload), _patch_extra_body():
                outcome = decomp.decompose_task(tid, author="me", dry_run=False)
        finally:
            for p in patches:
                p.stop()

        assert outcome.ok
        assert outcome.dry_run is False
        assert outcome.fanout is True
        assert outcome.child_ids is not None
        assert len(outcome.child_ids) == 1

        with kb.connect() as conn:
            root = kb.get_task(conn, tid)
            assert root.status == "todo"
            child = kb.get_task(conn, outcome.child_ids[0])
            assert child.status == "ready"
            assert child.assignee == "engineer"
            comments = kb.list_comments(conn, tid)
            events = kb.list_events(conn, tid)
            assert any("Decomposed into" in (c.body or "") for c in comments)
            assert any(ev.kind == "decomposed" for ev in events)
