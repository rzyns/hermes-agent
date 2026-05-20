"""Tests for the governance evaluate dry-run CLI surface (hermes_cli.governance).

TDD vertical slices — each test is a single behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import commands as cmd_reg


# ---------------------------------------------------------------------------
# Behavior 1: CLI help/registration recognizes the governance evaluate
# dry-run surface without invoking live integration.
# ---------------------------------------------------------------------------

class TestGovernanceCommandRegistry:
    def test_governance_command_def_exists(self):
        """COMMAND_REGISTRY contains a governance entry."""
        names = [c.name for c in cmd_reg.COMMAND_REGISTRY]
        assert "governance" in names

    def test_governance_has_evaluate_subcommand(self):
        """Governance command def lists 'evaluate' as a subcommand."""
        gov = next(c for c in cmd_reg.COMMAND_REGISTRY if c.name == "governance")
        assert "evaluate" in gov.subcommands

    def test_governance_parser_can_be_built(self):
        """build_parser() returns a parser that can parse evaluate --dry-run."""
        # This import should succeed once the module exists.
        from hermes_cli import governance

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        governance.build_parser(subparsers)

        args = parser.parse_args(["governance", "evaluate", "--dry-run", "--input", "/tmp/in.json", "--output-dir", "/tmp/out"])
        assert args.dry_run is True
        assert args.input == "/tmp/in.json"
        assert args.output_dir == "/tmp/out"


# ---------------------------------------------------------------------------
# Behavior 2: Given a minimal valid local fixture and output dir, command
# writes a decision/evaluation artifact and exits according to contract.
# ---------------------------------------------------------------------------

class TestGovernanceEvaluateDryRun:
    def test_valid_fixture_writes_artifact_and_exits_ok(self, tmp_path):
        """A valid input fixture results in exit code 0 and a decision artifact."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_OK

        artifact_path = out_dir / "decision.json"
        assert artifact_path.exists()
        artifact = json.loads(artifact_path.read_text())
        assert artifact["decision"] in {"allow", "deny"}
        assert "collector" in artifact
        assert "authorization" in artifact
        # Surface A: authorization must be explicitly False
        assert artifact["authorization"] is False

    def test_valid_blocked_fixture_exits_blocked(self, tmp_path):
        """A fixture with blocked status results in exit code 10."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "blocked"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_BLOCKED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert "authorization" in artifact


# ---------------------------------------------------------------------------
# Behavior 3: Missing/stale/malformed input fails closed with no
# authoritative decision artifact.
# ---------------------------------------------------------------------------

class TestGovernanceMissingInputFailsClosed:
    def test_missing_input_exits_fail_closed(self, tmp_path):
        """Missing input file returns EXIT_FAIL_CLOSED."""
        from hermes_cli import governance

        missing = tmp_path / "no_such_file.json"
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(missing),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_malformed_input_exits_fail_closed(self, tmp_path):
        """Malformed JSON returns EXIT_FAIL_CLOSED."""
        from hermes_cli import governance

        bad_fixture = tmp_path / "bad.json"
        bad_fixture.write_text("not json")
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(bad_fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False


# ---------------------------------------------------------------------------
# Behavior 4: Unknown/unmapped Kanban vocabulary fails closed / non-authoritative.
# Also verifies that repaired M3 source-derived vocabulary is accepted.
# ---------------------------------------------------------------------------

class TestGovernanceUnknownVocabularyFailsClosed:
    def test_unknown_status_fails_closed(self, tmp_path):
        """Unknown status is rejected at structure validation -> deny / exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "banana"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        # Unknown status is a structural failure in Surface A because the
        # minimal schema requires a *known* status; deny_unknown is reserved
        # for vocabulary on optional fields such as outcome.
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_unknown_outcome_fails_closed(self, tmp_path):
        """Unknown outcome returns EXIT_FAIL_CLOSED with deny_unknown."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "running", "outcome": "banana"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny_unknown"
        assert artifact["authorization"] is False
    # --- Repaired M3 source-derived status acceptance tests ---

    def test_status_triage_accepted(self, tmp_path):
        """M3-repaired status 'triage' is accepted (exit 0)."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "triage"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_OK
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "allow"

    def test_status_scheduled_accepted(self, tmp_path):
        """M3-repaired status 'scheduled' is accepted (exit 0)."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "scheduled"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_OK
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "allow"

    def test_status_review_accepted(self, tmp_path):
        """M3-repaired status 'review' is accepted (exit 0)."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "review"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_OK
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "allow"

    # --- Repaired M3 source-derived run outcome acceptance tests ---

    def test_outcome_gave_up_accepted(self, tmp_path):
        """M3-repaired outcome 'gave_up' is accepted (exit 0)."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "done", "outcome": "gave_up"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_OK
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "allow"

    def test_outcome_stale_accepted(self, tmp_path):
        """M3-repaired outcome 'stale' is accepted (exit 0)."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "done", "outcome": "stale"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_OK
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "allow"

    # --- Explicitly unsupported values must fail closed ---

    def test_outcome_operator_cancelled_fails_closed(self, tmp_path):
        """Unsupported outcome 'operator_cancelled' is deny_unknown / exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "done", "outcome": "operator_cancelled"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny_unknown"
        assert artifact["authorization"] is False

    def test_outcome_success_fails_closed(self, tmp_path):
        """Unsupported outcome 'success' is deny_unknown / exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "done", "outcome": "success"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny_unknown"
        assert artifact["authorization"] is False

    def test_status_closed_fails_closed(self, tmp_path):
        """Unsupported status 'closed' is deny / exit 20 (structural)."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "closed"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED
        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False


# ---------------------------------------------------------------------------
# Behavior 5: Command refuses to run without explicit dry-run/report-only mode.
# ---------------------------------------------------------------------------

class TestGovernanceDryRunGuard:
    def test_refuses_without_dry_run(self, tmp_path):
        """Missing --dry-run returns EXIT_FAIL_CLOSED."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=False,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

    def test_explicit_dry_run_required(self, tmp_path):
        """dry_run=False must gate even with valid input."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=False,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        # Must not return OK/10 — no authoritative artifact
        assert code == governance.EXIT_FAIL_CLOSED


# ---------------------------------------------------------------------------
# Behavior 6: Output artifact contains collector/CLI identity and clear
# non-authorization language.
# ---------------------------------------------------------------------------

class TestGovernanceArtifactIdentity:
    def test_artifact_contains_collector_identity(self, tmp_path):
        """Decision artifact includes COLLECTOR_IDENTITY."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        governance.cmd_evaluate(args)

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["collector"] == governance.COLLECTOR_IDENTITY

    def test_artifact_contains_non_authorization_flag(self, tmp_path):
        """authorization field is explicitly False."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({"status": "running"}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        governance.cmd_evaluate(args)

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert "authorization" in artifact
        assert artifact["authorization"] is False


# ---------------------------------------------------------------------------
# Behavior 7: Structurally malformed input fails closed (BLOCK-1 and BLOCK-2)
# ---------------------------------------------------------------------------

class TestGovernanceMalformedInputFailsClosed:
    def test_empty_object_fails_closed(self, tmp_path):
        """Empty JSON object {} is structurally invalid (missing status) -> exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps({}))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_json_array_fails_closed(self, tmp_path):
        """JSON array [] is type-mismatch -> fail-closed exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps([]))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_json_null_fails_closed(self, tmp_path):
        """JSON scalar null is type-mismatch -> fail-closed exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text("null")
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_json_int_fails_closed(self, tmp_path):
        """JSON scalar number 1 is type-mismatch -> fail-closed exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text("1")
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False

    def test_json_string_fails_closed(self, tmp_path):
        """JSON scalar string 'x' is type-mismatch -> fail-closed exit 20."""
        from hermes_cli import governance

        fixture = tmp_path / "fixture.json"
        fixture.write_text(json.dumps("x"))
        out_dir = tmp_path / "out"

        args = argparse.Namespace(
            dry_run=True,
            input=str(fixture),
            output_dir=str(out_dir),
            format="json",
            verbose=False,
        )
        code = governance.cmd_evaluate(args)
        assert code == governance.EXIT_FAIL_CLOSED

        artifact = json.loads((out_dir / "decision.json").read_text())
        assert artifact["decision"] == "deny"
        assert artifact["authorization"] is False
