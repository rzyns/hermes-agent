import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import (
    SelfHealPolicy,
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
)


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Map the target market and produce a decision memo.",
            workers=[
                SwarmWorkerSpec(
                    profile="researcher-a", title="Market scan", body="Find competitors"
                ),
                SwarmWorkerSpec(
                    profile="researcher-b",
                    title="Customer scan",
                    body="Find customer pains",
                ),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="intel",
            created_by="orchestrator",
        )

        root = kb.get_task(conn, created.root_id)
        workers = [kb.get_task(conn, tid) for tid in created.worker_ids]
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        assert root.status == "done"
        assert root.assignee == "orchestrator"
        assert [task.status for task in workers] == ["ready", "ready"]
        assert [task.assignee for task in workers] == ["researcher-a", "researcher-b"]
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(created.worker_ids)
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        assert all(created.root_id in (task.body or "") for task in workers)
    finally:
        conn.close()


def test_swarm_blackboard_merges_structured_updates(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Collect evidence.",
            workers=[
                SwarmWorkerSpec(
                    profile="researcher", title="Evidence", body="Find proof"
                )
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        post_blackboard_update(
            conn,
            created.root_id,
            author="researcher",
            key="sources",
            value=["https://example.com/a"],
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="reviewer",
            key="risks",
            value={"missing_primary_source": True},
        )

        board = latest_blackboard(conn, created.root_id)
        assert board["sources"] == ["https://example.com/a"]
        assert board["risks"] == {"missing_primary_source": True}
        assert board["_authors"]["sources"] == "researcher"
    finally:
        conn.close()


def test_swarm_verifier_and_synthesis_are_dependency_gated(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Research two branches then verify and synthesize.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="b", title="Branch B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        kb.complete_task(
            conn,
            created.worker_ids[0],
            summary="A done",
            metadata={"confidence": 0.8},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "todo"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "ready"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.synthesizer_id).status == "ready"
    finally:
        conn.close()


def test_preflight_fail_raises_on_missing_skill(tmp_path, monkeypatch):
    """When policy=fail, create_swarm aborts before cards exist."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    skills = tmp_path / "hermes" / "skills" / "devops" / "kanban-worker"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: kanban-worker\n---\n")

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError) as exc_info:
            create_swarm(
                conn,
                goal="Test fail-fast.",
                workers=[
                    SwarmWorkerSpec(
                        profile="fake-profile",
                        title="Worker",
                        body="Do work",
                        skills=["definitely-missing-skill"],
                    ),
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                self_heal=SelfHealPolicy(mode="fail"),
            )
        assert "definitely-missing-skill" in str(exc_info.value)
        # No swarm root should have been created.
        assert len(kb.list_tasks(conn)) == 0
    finally:
        conn.close()


def test_preflight_drop_removes_missing_skills(tmp_path, monkeypatch):
    """When policy=drop, unavailable skills are stripped and the worker is created."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    skills = tmp_path / "hermes" / "skills" / "devops" / "kanban-worker"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: kanban-worker\n---\n")
    available = tmp_path / "hermes" / "skills" / "research" / "available-skill"
    available.mkdir(parents=True)
    (available / "SKILL.md").write_text("---\nname: available-skill\n---\n")

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Test drop.",
            workers=[
                SwarmWorkerSpec(
                    profile="fake-profile",
                    title="Worker",
                    body="Do work",
                    skills=["available-skill", "missing-skill"],
                ),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            self_heal=SelfHealPolicy(mode="drop"),
        )
        worker = kb.get_task(conn, created.worker_ids[0])
        assert worker.skills == ["available-skill"]
        board = latest_blackboard(conn, created.root_id)
        assert board["preflight_skill_check"][0]["missing_skills"] == ["missing-skill"]
    finally:
        conn.close()


def test_preflight_repair_creates_repair_card(tmp_path, monkeypatch):
    """When policy=repair, a missing skill produces a blocked repair card."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    skills = tmp_path / "hermes" / "skills" / "devops" / "kanban-worker"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: kanban-worker\n---\n")

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Test repair.",
            workers=[
                SwarmWorkerSpec(
                    profile="fake-profile",
                    title="Worker",
                    body="Do work",
                    skills=["missing-skill"],
                ),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            self_heal=SelfHealPolicy(mode="repair", healer_profile="ops"),
        )
        assert len(created.worker_ids) == 0
        assert len(created.repair_ids) == 1
        repair = kb.get_task(conn, created.repair_ids[0])
        assert repair.assignee == "ops"
        assert repair.status == "blocked"
        assert "missing-skill" in (repair.body or "")
        # Verifier still depends on root, since no workers were created.
        assert kb.parent_ids(conn, created.verifier_id) == []
    finally:
        conn.close()


def test_classify_worker_log_failure_detects_missing_skill():
    result = kb.classify_worker_log_failure(
        "Warning: Unknown toolsets: memory_wiki, messaging\n"
        "Error: Unknown skill(s): oss-substrate-audit, kanban-evidence-audit\n"
    )
    assert result is not None
    assert result["kind"] == "missing_skill"
    assert result["missing_skills"] == ["oss-substrate-audit", "kanban-evidence-audit"]


def test_classify_worker_log_failure_returns_none_for_unmatched_log():
    assert kb.classify_worker_log_failure("Some random worker output") is None


def test_find_swarm_root_walks_ancestor_chain(tmp_path):
    """_find_swarm_root finds the root through worker -> verifier chains."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Ancestor chain test.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Worker A", body="A"),
                SwarmWorkerSpec(profile="b", title="Worker B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        # Verifier is a child of workers, not the root.
        verifier_id = created.verifier_id
        root_id = created.root_id
        assert kb._find_swarm_root(conn, verifier_id) == root_id
        # Worker is a direct child of root.
        assert kb._find_swarm_root(conn, created.worker_ids[0]) == root_id
        # Synthesizer is a child of verifier.
        assert kb._find_swarm_root(conn, created.synthesizer_id) == root_id
        # Non-swarm task returns None.
        other = kb.create_task(conn, title="other")
        assert kb._find_swarm_root(conn, other) is None
    finally:
        conn.close()


def test_create_swarm_repair_task_finds_verifier_ancestor_root(tmp_path, monkeypatch):
    """Auto-repair works for verifiers that are not direct children of the root."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    skills = tmp_path / "hermes" / "skills" / "devops" / "kanban-worker"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: kanban-worker\n---\n")

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Verifier ancestor test.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Worker A", body="A"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        verifier_id = created.verifier_id
        root_id = created.root_id

        # Write a fake worker log for the verifier with a missing skill.
        log_path = kb.worker_log_path(verifier_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Warning: Unknown toolsets: memory_wiki, messaging\n"
            "Error: Unknown skill(s): requesting-code-review\n",
            encoding="utf-8",
        )

        repair_id = kb.create_swarm_repair_task(
            conn, verifier_id, healer_profile="ops", created_by="self-healer"
        )
        assert repair_id is not None
        repair = kb.get_task(conn, repair_id)
        assert repair is not None
        assert repair.assignee == "ops"
        assert repair.status == "blocked"
        assert root_id in kb.parent_ids(conn, repair_id)
        # Failure classification is preserved in body.
        assert "requesting-code-review" in (repair.body or "")
        # Idempotency: a second call does not create a duplicate.
        assert (
            kb.create_swarm_repair_task(conn, verifier_id, healer_profile="ops") is None
        )
    finally:
        conn.close()


def test_create_swarm_repair_task_idempotent_across_calls(tmp_path, monkeypatch):
    """Repeated calls for the same crashed worker create only one repair card."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    skills = tmp_path / "hermes" / "skills" / "devops" / "kanban-worker"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: kanban-worker\n---\n")

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Idempotency test.",
            workers=[SwarmWorkerSpec(profile="w", title="Worker", body="W")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        worker_id = created.worker_ids[0]
        root_id = created.root_id

        log_path = kb.worker_log_path(worker_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Error: Unknown skill(s): some-missing-skill\n",
            encoding="utf-8",
        )

        first = kb.create_swarm_repair_task(conn, worker_id, healer_profile="ops")
        second = kb.create_swarm_repair_task(conn, worker_id, healer_profile="ops")
        assert first is not None
        assert second is None
        # Only one repair card exists under the root.
        children = kb.child_ids(conn, root_id)
        repair_children = [
            c
            for c in children
            if (kb.get_task(conn, c).title or "").startswith("Repair:")
        ]
        assert len(repair_children) == 1
    finally:
        conn.close()


def test_create_swarm_repair_task_no_repair_for_unmatched_failure(tmp_path):
    """Operational failures without a known classification do not create cards."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Unmatched failure test.",
            workers=[SwarmWorkerSpec(profile="w", title="Worker", body="W")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        worker_id = created.worker_ids[0]
        log_path = kb.worker_log_path(worker_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Some random worker output", encoding="utf-8")
        assert kb.create_swarm_repair_task(conn, worker_id) is None
    finally:
        conn.close()
