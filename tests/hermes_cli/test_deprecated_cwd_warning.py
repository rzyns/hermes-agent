"""Targeted tests for warn_deprecated_cwd_env_vars() hermetic warning behavior.

These exercise:
- Correct warning emission when deprecated env vars are set.
- Skipping when config.yaml has explicit terminal.cwd.
- Warning visibility survives config load failure (no silent swallow).
- Import-time leakage precondition: importing gateway.run must NOT emit warning.
- End-to-end CLI governance dispatch must NOT leak warning into stderr.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import config as config_module


@pytest.fixture
def hermes_main():
    """Path to the repo root for subprocess invocation."""
    repo = Path(__file__).resolve().parents[2]
    return repo


class TestDeprecatedCwdWarningEmission:
    """Unit-level tests for warn_deprecated_cwd_env_vars()."""

    def test_warn_emits_stderr_when_messaging_cwd_set(self, capsys, monkeypatch):
        monkeypatch.setenv("MESSAGING_CWD", "/some/path")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        config_module.warn_deprecated_cwd_env_vars(config={})
        captured = capsys.readouterr()
        assert "MESSAGING_CWD=/some/path" in captured.err
        assert "deprecated" in captured.err.lower()

    def test_warn_emits_stderr_when_terminal_cwd_in_env_not_config(
        self, capsys, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_CWD", "/other/path")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        config_module.warn_deprecated_cwd_env_vars(config={})
        captured = capsys.readouterr()
        assert "TERMINAL_CWD=/other/path" in captured.err
        assert "deprecated" in captured.err.lower()

    def test_warn_skips_terminal_cwd_when_config_has_explicit_cwd(
        self, capsys, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_CWD", "/other/path")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        config_module.warn_deprecated_cwd_env_vars(
            config={"terminal": {"cwd": "/explicit/path"}}
        )
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_warn_skips_when_no_deprecated_vars(self, capsys, monkeypatch):
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        config_module.warn_deprecated_cwd_env_vars(config={"terminal": {}})
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_warn_survives_config_load_failure(self, capsys, monkeypatch):
        """If load_config() raises, the deprecation warning should still fire.

        Regression against silent ``except Exception: return``.
        """
        monkeypatch.setenv("MESSAGING_CWD", "/some/path")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        with patch.object(
            config_module,
            "load_config",
            side_effect=RuntimeError("simulated config load failure"),
        ):
            config_module.warn_deprecated_cwd_env_vars()

        captured = capsys.readouterr()
        assert "MESSAGING_CWD=/some/path" in captured.err


class TestGatewayRunnerImportNoLeak:
    """Integration tests reproducing the import-time leakage precondition."""

    def test_import_gatewayrunner_does_not_emit_deprecated_warning(
        self, tmp_path, hermes_main
    ):
        """Reproduces the original pre-fix leak: importing gateway.run with deprecated
        env vars set must NOT emit the warning at import time.

        Before fix: gateway/run.py called warn_deprecated_cwd_env_vars() at module scope.
        After fix: the call is inside GatewayRunner.__init__.
        """
        env = os.environ.copy()
        env["MESSAGING_CWD"] = "/tmp/fake_messaging_cwd"
        env["TERMINAL_CWD"] = "/tmp/fake_terminal_cwd"

        trigger = tmp_path / "trigger_import.py"
        trigger.write_text(
            f"""\
import sys
sys.path.insert(0, {str(hermes_main)!r})
import gateway.run
print(gateway.run.GatewayRunner.__name__)
"""
        )

        result = subprocess.run(
            [sys.executable, str(trigger)],
            capture_output=True,
            text=True,
            cwd=str(hermes_main),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "GatewayRunner" in result.stdout
        assert "MESSAGING_CWD" not in result.stderr
        assert "TERMINAL_CWD" not in result.stderr
        assert "Deprecated" not in result.stderr

    def test_governance_cli_does_not_leak_with_deprecated_env_set(
        self, tmp_path, hermes_main
    ):
        """End-to-end: governance evaluate --dry-run with deprecated env set must
        complete without leaking warning into stderr.
        """
        fixture = tmp_path / "in.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        env = os.environ.copy()
        env["MESSAGING_CWD"] = "/tmp/fake_messaging_cwd"
        env["TERMINAL_CWD"] = "/tmp/fake_terminal_cwd"

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
            env=env,
        )
        assert result.returncode == 0
        assert "Deprecated .env settings detected" not in result.stderr
        assert "MESSAGING_CWD" not in result.stderr
        assert "TERMINAL_CWD" not in result.stderr
