from __future__ import annotations

import json
from pathlib import Path

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
