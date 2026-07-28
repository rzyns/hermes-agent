"""The unblock-loop breaker must summon a human, not the auto-decomposer.

The defect, observed on a live installation 2026-07-28.

``block_task`` implements an unblock-loop breaker: once a task has been
blocked / unblocked / re-blocked for the same cause ``BLOCK_RECURRENCE_LIMIT``
times, it stops routing the task to ``blocked`` and instead sets
``status='triage'``. The comment at the escalation site says exactly what that
is for:

    Loop detected — stop letting the unblocker spin this task. Route
    to triage for a human-in-the-loop decision instead of blocked.

But ``triage`` is also the intake queue of the **auto-decomposer**
(``kanban_decompose.list_triage_ids`` → ``decompose_task``, run every gateway
tick when ``kanban.auto_decompose`` is on, which is the default). Nothing
distinguishes a card that a human dropped into Triage as a rough idea from a
card the loop breaker escalated *because it needs a human*.

So the escalation path designed to summon a human hands the card to an LLM
instead. The decomposer fans it into children and flips the root to ``todo``,
the human is never asked, and the next time the root re-blocks the cycle
repeats.

Live consequence: root card ``t_b1339fbf`` was decomposed three times into six
redundant children each pass — including a *mutating* commit card aimed at a
deliberately paused lane — while it was waiting on a human release decision it
had already correctly asked for.

These tests pin the boundary: the loop breaker keeps escalating (that part is
correct and must not regress), but an escalated card must not be offered to the
auto-decomposer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as kd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _escalate_to_triage(conn, title="needs a human"):
    """Drive a task through the loop breaker until it lands in triage."""
    tid = _running_task(conn, title=title)
    kb.block_task(conn, tid, reason="need a human decision", kind="needs_input")
    kb.unblock_task(conn, tid)
    _make_running_again(conn, tid)
    kb.block_task(conn, tid, reason="still need a human decision", kind="needs_input")
    task = kb.get_task(conn, tid)
    assert task.status == "triage", "precondition: loop breaker should escalate to triage"
    assert task.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
    return tid


def test_escalation_still_routes_to_triage(kanban_home: Path) -> None:
    """Guard rail: do not 'fix' this by weakening the loop breaker itself."""
    with kb.connect_closing() as conn:
        tid = _escalate_to_triage(conn)
        task = kb.get_task(conn, tid)

    assert task.status == "triage"
    assert task.block_kind == "needs_input"
    assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT


def test_loop_escalated_task_is_not_offered_to_auto_decomposer(kanban_home: Path) -> None:
    """The defect. An escalated card is waiting for a human, not for an LLM."""
    with kb.connect_closing() as conn:
        tid = _escalate_to_triage(conn)

    offered = kd.list_triage_ids()

    assert tid not in offered, (
        f"task {tid} was escalated to triage BY the unblock-loop breaker, "
        "explicitly to force a human-in-the-loop decision, and the auto-decomposer "
        "is being offered it as if it were a fresh idea to fan out"
    )


def test_fresh_triage_task_is_still_offered_to_auto_decomposer(kanban_home: Path) -> None:
    """The feature must keep working for genuine triage input."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="rough idea a human dropped in Triage",
            assignee="worker",
            triage=True,
        )
        assert kb.get_task(conn, tid).status == "triage"

    offered = kd.list_triage_ids()

    assert tid in offered, "a genuine triage card must still be auto-decomposable"


def test_both_kinds_present_only_the_fresh_one_is_offered(kanban_home: Path) -> None:
    """The two must be distinguishable when they coexist on one board."""
    with kb.connect_closing() as conn:
        escalated = _escalate_to_triage(conn, title="escalated")
        fresh = kb.create_task(
            conn, title="fresh idea", assignee="worker", triage=True,
        )

    offered = set(kd.list_triage_ids())

    assert fresh in offered
    assert escalated not in offered


def test_decompose_task_refuses_a_loop_escalated_task(kanban_home: Path) -> None:
    """Defence in depth: a direct call must refuse too, not just the listing.

    ``list_triage_ids`` is the gateway's intake, but ``decompose_task`` is also
    reachable directly (CLI, dashboard, tools). Filtering only the listing would
    leave the door open.
    """
    with kb.connect_closing() as conn:
        tid = _escalate_to_triage(conn)

    outcome = kd.decompose_task(tid, author="auto-decomposer")

    assert not outcome.ok, "decompose_task should refuse a loop-escalated card"
    assert "human" in (outcome.reason or "").lower() or "escalat" in (outcome.reason or "").lower(), (
        f"refusal reason should explain why; got: {outcome.reason!r}"
    )
