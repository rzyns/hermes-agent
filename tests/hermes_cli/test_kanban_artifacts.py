from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_artifacts as ka


def test_write_artifact_manifest_records_workspace_relative_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = workspace / "report.md"
    report.write_text("# Findings\n", encoding="utf-8")

    manifest = ka.write_artifact_manifest(
        task_id="t_deadbeef",
        board="agent-research-intake",
        workspace_path=workspace,
        artifacts=[report],
        metadata={"source_ref": "attention-intake/t_source"},
    )

    assert manifest["valid"] is True
    assert manifest["manifest_path"] == str(workspace / ka.MANIFEST_FILENAME)
    assert manifest["artifacts"] == [{
        "path": str(report.resolve(strict=False)),
        "raw_path": str(report),
        "relative_path": "report.md",
        "exists": True,
        "is_file": True,
        "size_bytes": len("# Findings\n"),
        "in_workspace": True,
    }]
    written = json.loads((workspace / ka.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert written["metadata"]["source_ref"] == "attention-intake/t_source"


def test_build_artifact_manifest_flags_missing_or_external_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "outside.txt"
    external.write_text("outside", encoding="utf-8")
    missing = workspace / "missing.txt"

    manifest = ka.build_artifact_manifest(
        task_id="t_deadbeef",
        workspace_path=workspace,
        artifacts=[external, missing],
    )

    assert manifest["valid"] is False
    assert manifest["artifacts"][0]["exists"] is True
    assert manifest["artifacts"][0]["in_workspace"] is False
    assert manifest["artifacts"][1]["exists"] is False
    assert manifest["artifacts"][1]["in_workspace"] is True


def test_post_publish_readback_gate_catches_hash_mismatch(tmp_path):
    """The attachment-insertion readback gate must re-hash stored files and
    reject a manifest whose recorded content_hash no longer matches disk bytes."""
    import hashlib
    from hermes_cli import kanban_db as kb

    stored = tmp_path / "report.md"
    stored.write_bytes(b"original bytes")
    expected = hashlib.sha256(b"original bytes").hexdigest()
    manifest = [
        {
            "logical_name": "report.md",
            "source_path": str(stored),
            "stored_path": str(stored),
            "size": len(b"original bytes"),
            "content_hash": expected,
        }
    ]
    assert kb._verify_published_artifact_manifest(manifest) == []

    # Simulate corruption/overwrite after insertion.
    stored.write_bytes(b"tampered bytes")
    failures = kb._verify_published_artifact_manifest(manifest)
    assert failures
    assert "post-publish hash mismatch" in failures[0]


def test_post_publish_readback_wired_into_complete_task(monkeypatch, tmp_path):
    """The post-insertion readback gate must run inside complete_task's
    artifact publication path. Removing the call to
    _verify_published_artifact_manifest would let corrupted attachments
    slip through; this test proves the wiring is live by corrupting the
    stored file after the first completion and forcing a re-finalization
    that re-enters the publish/readback path.

    Because _publish_completion_artifact_manifest de-duplicates by content
    hash, the same source bytes would reuse the already-corrupted stored
    path. We work around that by finalizing a row whose
    published_artifacts_json already points at the corrupted stored path.
    """
    import hashlib
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workspaces_root = tmp_path / "kanban" / "workspaces"
    workspaces_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces_root))

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="post-publish wiring test", assignee="worker",
            body="x", workspace_kind="scratch", workspace_path=str(workspaces_root / "t_post")
        )
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    workspace = workspaces_root / "t_post"
    workspace.mkdir()
    report = workspace / "report.md"
    report.write_bytes(b"original bytes")

    # First completion: artifact is published successfully.
    conn2 = kb.connect()
    try:
        result = kb.complete_task(
            conn2, tid, summary="first",
            metadata={"artifacts": [str(report)]},
        )
        assert result.ok is True
    finally:
        conn2.close()

    # Locate the durable stored attachment and corrupt it.
    attachment_dir = tmp_path / "kanban" / "attachments" / tid
    stored = next(attachment_dir.rglob("report.md"))
    stored.write_bytes(b"tampered bytes")

    # Reset the completion row to 'committed' so the next finalize call
    # re-enters the publish/readback path. Inject a manifest that already
    # names the corrupted stored path with the *original* hash so the
    # readback detects the mismatch.
    good_hash = hashlib.sha256(b"original bytes").hexdigest()
    conn3 = kb.connect()
    try:
        run_id = kb._latest_terminal_run_id(conn3, tid)
        conn3.execute(
            """
            UPDATE task_completion_results
               SET status = 'committed',
                   finalized_at = NULL,
                   published_artifacts_json = ?
             WHERE task_id = ? AND run_id = ?
            """,
            (
                json.dumps([{
                    "source_path": str(stored),
                    "stored_path": str(stored),
                    "logical_name": "report.md",
                    "size": stored.stat().st_size,
                    "content_hash": good_hash,
                }]),
                tid,
                run_id,
            ),
        )
        conn3.commit()
    finally:
        conn3.close()

    conn4 = kb.connect()
    try:
        with pytest.raises(kb.ArtifactPreservationError) as exc_info:
            kb._finalize_completion_result(
                conn4, tid, run_id, None,
                review_pending_ids=(), summary="first", result=None,
            )
        assert "post-publish hash mismatch" in str(exc_info.value)
    finally:
        conn4.close()


def test_post_publish_readback_gate_flags_missing_file(tmp_path):
    """The attachment-insertion readback gate must flag a stored_path that no
    longer exists on disk."""
    from hermes_cli import kanban_db as kb

    stored = tmp_path / "gone.txt"
    manifest = [
        {
            "logical_name": "gone.txt",
            "source_path": str(stored),
            "stored_path": str(stored),
            "size": 0,
            "content_hash": "0" * 64,
        }
    ]
    failures = kb._verify_published_artifact_manifest(manifest)
    assert failures
    assert "published artifact missing on disk" in failures[0]
