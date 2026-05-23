"""End-to-end CLI tests for ``hermes governance evaluate --dry-run``.

These exercise the full parse-dispatch-evaluate path via subprocess
invocation of ``python -m hermes_cli.main governance evaluate ...``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def hermes_main():
    """Path to the repo root for subprocess invocation via -m hermes_cli.main."""
    repo = Path(__file__).resolve().parents[2]
    return repo


class TestGovernanceCliEndToEnd:
    def test_cli_parser_accepts_evaluate_dry_run(self, tmp_path, hermes_main):
        """The real argparse plumbing accepts governance evaluate --dry-run."""
        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--dry-run",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        # Exit code 0 means evaluation succeeded (but policy is non-authorization).
        assert result.returncode == 0
        assert (out_dir / "decision.json").exists()

    def test_cli_dispatch_returns_blocked_for_blocked_input(self, tmp_path, hermes_main):
        """Full path returns exit code 10 for a blocked-status fixture."""
        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps({"status": "blocked"}))
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--dry-run",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        assert result.returncode == 10

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_cli_dispatch_fails_closed_without_dry_run(self, tmp_path, hermes_main):
        """Full path returns 20 when --dry-run is omitted."""
        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        assert result.returncode == 20
        assert "--dry-run is required" in result.stderr

    def test_cli_artifact_contains_collector_identity(self, tmp_path, hermes_main):
        """The full-path decision artifact contains collector id."""
        from hermes_cli import governance

        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--dry-run",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        assert result.returncode == 0

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact.get("collector") == governance.COLLECTOR_IDENTITY
        assert artifact.get("authorization") is False

    def test_cli_empty_object_exits_20(self, tmp_path, hermes_main):
        """Full path: empty JSON object {} -> exit 20 fail-closed."""
        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps({}))
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--dry-run",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        assert result.returncode == 20

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_cli_json_array_exits_20(self, tmp_path, hermes_main):
        """Full path: JSON array [] -> exit 20 fail-closed (not 1)."""
        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps([]))
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--dry-run",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        assert result.returncode == 20

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_cli_json_null_exits_20(self, tmp_path, hermes_main):
        """Full path: JSON scalar null -> exit 20 fail-closed."""
        fixture = tmp_path / "in.json"
        fixture.write_text("null")
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "governance", "evaluate",
                "--dry-run",
                "--input", str(fixture),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
        )
        assert result.returncode == 20

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False
