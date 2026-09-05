import pytest
import time
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_swarm import (
    SelfHealPolicy,
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
    preflight_worker_skills,
)


def _write_skill(root, name):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kbc.connect(tmp_path / "kanban.db")
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

        assert root is not None
        assert all(task is not None for task in workers)
        workers = [task for task in workers if task is not None]
        assert verifier is not None
        assert synthesizer is not None
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


def test_create_swarm_graph_is_atomic_and_rolls_back_partial_build(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    db_path = tmp_path / "kanban.db"
    writer = kbc.connect(db_path)
    reader = kbc.connect(db_path)
    original_create = kb.create_task
    original_complete = kb.complete_task
    calls = 0

    def observed_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        task_id = original_create(*args, **kwargs)
        if calls == 1:
            # Releasing the nested create_task savepoint must not expose the
            # root before the whole graph's outer transaction commits.
            visible = reader.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            assert visible == 0
        if calls == 3:
            raise RuntimeError("synthetic graph-construction failure")
        return task_id

    monkeypatch.setattr(kb, "create_task", observed_create)
    try:
        with pytest.raises(RuntimeError, match="synthetic graph-construction failure"):
            create_swarm(
                writer,
                goal="Build atomically",
                workers=[
                    SwarmWorkerSpec(profile="worker-a", title="A", body="A"),
                    SwarmWorkerSpec(profile="worker-b", title="B", body="B"),
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
            )
        assert writer.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert reader.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

        monkeypatch.setattr(kb, "create_task", original_create)
        import hermes_cli.kanban_swarm as ks

        original_activate = ks._activate_root_inline
        monkeypatch.setattr(
            ks, "_activate_root_inline", lambda *args, **kwargs: False
        )
        with pytest.raises(RuntimeError, match="could not activate"):
            create_swarm(
                writer,
                goal="Fail activation atomically",
                workers=[
                    SwarmWorkerSpec(profile="worker-a", title="A", body="A"),
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
            )
        assert writer.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert reader.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

        hooks: list[tuple[str, bool]] = []
        monkeypatch.setattr(ks, "_activate_root_inline", original_activate)
        monkeypatch.setattr(
            kb,
            "_fire_kanban_lifecycle_hook",
            lambda event, *_args, **_kwargs: hooks.append(
                (event, writer.in_transaction)
            ),
        )
        create_swarm(
            writer,
            goal="Commit before lifecycle hook",
            workers=[SwarmWorkerSpec(profile="worker-a", title="A", body="A")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        assert hooks == [("kanban_task_completed", False)]
    finally:
        reader.close()
        writer.close()


def test_plain_write_txn_nesting_raises_and_allow_nested_composes(tmp_path):
    """B1 regression: nesting is explicit opt-in, never silent.

    Plain ``write_txn`` inside an open transaction must raise loudly (the
    historical invariant). ``allow_nested=True`` composes via a savepoint,
    and an outer rollback discards the inner work without any post-commit
    side effects having fired (the workspace directory survives).
    """
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        workspace = tmp_path / "scratch-ws"
        workspace.mkdir()
        tid = kb.create_task(conn, title="ws task", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                (str(workspace), tid),
            )

        # 1) Plain nesting raises loudly.
        with pytest.raises(RuntimeError, match="already inside a transaction"):
            with kb.write_txn(conn):
                with kb.write_txn(conn):
                    pass
        assert not conn.in_transaction

        # 2) allow_nested composes; outer rollback discards inner work
        #    and no side effects (workspace cleanup) fired meanwhile.
        with pytest.raises(RuntimeError, match="outer failure"):
            with kb.write_txn(conn):
                with kb.write_txn(conn, allow_nested=True):
                    conn.execute(
                        "UPDATE tasks SET status = 'done' WHERE id = ?", (tid,)
                    )
                    kb._append_event(conn, tid, "completed", {"result_len": 0})
                # Inner savepoint released, but the outer txn now fails.
                raise RuntimeError("outer failure")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"  # inner 'done' flip was discarded
        assert not any(
            e.kind == "completed" for e in kb.list_events(conn, tid)
        )
        assert workspace.is_dir()  # no _cleanup_workspace side effect fired
    finally:
        conn.close()


def test_swarm_blackboard_merges_structured_updates(tmp_path):
    conn = kbc.connect(tmp_path / "kanban.db")
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
    conn = kbc.connect(tmp_path / "kanban.db")
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
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert verifier is not None
        assert synthesizer is not None
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert verifier is not None
        assert synthesizer is not None
        assert verifier.status == "ready"
        assert synthesizer.status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert synthesizer is not None
        assert synthesizer.status == "ready"
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


def test_preflight_fail_checks_generated_gate_skills_before_creating_cards(tmp_path, monkeypatch):
    """Fail-fast preflight covers generated verifier/synthesizer skills too."""
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "skills").mkdir(parents=True)
    # The generated gate profiles exist, but their profile-local skill trees do
    # not contain the helper skills that the swarm helper would otherwise force.
    for profile in ("reviewer", "writer"):
        profile_home = hermes_home / "profiles" / profile
        (profile_home / "skills").mkdir(parents=True)
        (profile_home / "config.yaml").write_text(
            "skills:\n  external_dirs: []\n",
            encoding="utf-8",
        )

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError) as exc_info:
            create_swarm(
                conn,
                goal="Generated gate preflight should fail before DB writes.",
                workers=[SwarmWorkerSpec(profile="default", title="Worker", body="Do work")],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                self_heal=SelfHealPolicy(mode="fail"),
            )
        msg = str(exc_info.value)
        assert "reviewer/Verify swarm outputs" in msg
        assert "requesting-code-review" in msg
        assert "writer/Synthesize swarm outputs" in msg
        assert "humanizer" in msg
        assert len(kb.list_tasks(conn)) == 0
    finally:
        conn.close()


def test_preflight_uses_profile_external_dirs_not_dispatcher_home(tmp_path, monkeypatch):
    """Skill preflight matches the profile-scoped HERMES_HOME used by workers."""
    hermes_home = tmp_path / "hermes"
    external_skills = tmp_path / "external-skills"
    _write_skill(external_skills, "profile-visible-skill")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "skills").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_skills}\n",
        encoding="utf-8",
    )
    profile_home = hermes_home / "profiles" / "worker-profile"
    (profile_home / "skills").mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "skills:\n  external_dirs: []\n",
        encoding="utf-8",
    )

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError) as exc_info:
            create_swarm(
                conn,
                goal="Profile-specific preflight should not trust default external dirs.",
                workers=[
                    SwarmWorkerSpec(
                        profile="worker-profile",
                        title="Worker",
                        body="Do work",
                        skills=["profile-visible-skill"],
                    ),
                ],
                verifier_assignee="default",
                synthesizer_assignee="default",
                self_heal=SelfHealPolicy(mode="fail"),
            )
        assert "worker-profile/Worker" in str(exc_info.value)
        assert "profile-visible-skill" in str(exc_info.value)
        assert len(kb.list_tasks(conn)) == 0
    finally:
        conn.close()


def test_preflight_resolves_profile_external_dirs(tmp_path, monkeypatch):
    """The resolver honors the real config key: skills.external_dirs."""
    hermes_home = tmp_path / "hermes"
    external_skills = tmp_path / "external-skills"
    _write_skill(external_skills, "profile-visible-skill")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_home = hermes_home / "profiles" / "worker-profile"
    (profile_home / "skills").mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_skills}\n",
        encoding="utf-8",
    )

    available, missing = preflight_worker_skills(
        SwarmWorkerSpec(
            profile="worker-profile",
            title="Worker",
            body="Do work",
            skills=["profile-visible-skill"],
        )
    )
    assert available == ["profile-visible-skill"]
    assert missing == []


def test_preflight_repair_drops_missing_generated_gate_skills(tmp_path, monkeypatch):
    """Repair/drop policy must not create verifier/synth cards that will crash."""
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "skills").mkdir(parents=True)
    for profile in ("reviewer", "writer"):
        profile_home = hermes_home / "profiles" / profile
        (profile_home / "skills").mkdir(parents=True)
        (profile_home / "config.yaml").write_text(
            "skills:\n  external_dirs: []\n",
            encoding="utf-8",
        )

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Repair policy should keep generated gates dispatchable.",
            workers=[SwarmWorkerSpec(profile="default", title="Worker", body="Do work")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            self_heal=SelfHealPolicy(mode="repair", healer_profile="ops"),
        )

        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert verifier is not None
        assert synthesizer is not None
        assert verifier.skills == []
        assert synthesizer.skills == []
        board = latest_blackboard(conn, created.root_id)
        gate_entries = [
            entry for entry in board["preflight_skill_check"]
            if entry.get("generated_gate") is True
        ]
        assert gate_entries == [
            {
                "profile": "reviewer",
                "title": "Verify swarm outputs",
                "missing_skills": ["requesting-code-review"],
                "action": "drop_generated_gate_skill",
                "generated_gate": True,
            },
            {
                "profile": "writer",
                "title": "Synthesize swarm outputs",
                "missing_skills": ["humanizer"],
                "action": "drop_generated_gate_skill",
                "generated_gate": True,
            },
        ]
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


def test_classify_worker_log_failure_checks_startup_error_at_log_head():
    result = kb.classify_worker_log_failure(
        "Error: Unknown skill(s): requesting-code-review\n"
        + ("later verbose output\n" * 1000)
    )
    assert result is not None
    assert result["kind"] == "missing_skill"
    assert result["missing_skills"] == ["requesting-code-review"]


def test_classify_worker_log_failure_does_not_consume_next_error_line():
    result = kb.classify_worker_log_failure(
        "Error: Unknown skill(s): requesting-code-review\n"
        "Error: Unknown skill(s): humanizer\n"
    )
    assert result is not None
    assert result["kind"] == "missing_skill"
    assert result["missing_skills"] == ["requesting-code-review"]


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


def test_dispatch_auto_repair_uses_explicit_board_logs(tmp_path, monkeypatch):
    """Gateway-style multi-board dispatch must classify crashes on that board."""
    import hermes_cli.kanban_db as _kb

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "kanban:\n  self_heal:\n    enabled: true\n    healer_profile: ops\n",
        encoding="utf-8",
    )

    kb.create_board("other-board")
    kb.create_board("swarm-board")
    kb.set_current_board("other-board")

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(_kb, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 1))

    with kb.connect(board="swarm-board") as conn:
        created = create_swarm(
            conn,
            goal="Board-aware auto repair.",
            workers=[SwarmWorkerSpec(profile="default", title="Worker", body="Do work")],
            verifier_assignee="default",
            synthesizer_assignee="default",
        )
        verifier_id = created.verifier_id
        host = _kb._claimer_id().split(":", 1)[0]
        conn.execute(
            "UPDATE tasks SET status='running', worker_pid=?, claim_lock=?, "
            "started_at=?, consecutive_failures=? WHERE id=?",
            (424242, f"{host}:worker", int(time.time()) - 3600, 2, verifier_id),
        )
        log_path = kb.worker_log_path(verifier_id, board="swarm-board")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Error: Unknown skill(s): requesting-code-review\n",
            encoding="utf-8",
        )
        conn.commit()

        result = kb.dispatch_once(conn, dry_run=True, board="swarm-board")
        assert verifier_id in result.auto_blocked
        repair_children = []
        for child_id in kb.child_ids(conn, created.root_id):
            child = kb.get_task(conn, child_id)
            assert child is not None
            if (child.title or "").startswith("Repair:"):
                repair_children.append(child_id)
        assert len(repair_children) == 1
        repair = kb.get_task(conn, repair_children[0])
        assert repair is not None
        assert repair.assignee == "ops"
        assert "requesting-code-review" in (repair.body or "")
