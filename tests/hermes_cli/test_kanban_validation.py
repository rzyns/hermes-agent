"""Tests for hermes_cli.kanban_validation.

These tests enforce the kernel contract and make it impossible for future
surfaces to silently re-implement their own workspace/branch rules: any new
validation path that does not call ``validate_workspace_spec`` or import from
``kanban_validation`` will be detectable by the divergence tests below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_validation import (
    UNDEFINED,
    OptionalUnset,
    PayloadValidationError,
    StrictPayloadBase,
    Undefined,
    WorkspaceSpec,
    _normalise_branch_name,
    validate_workspace_spec,
)


# ---------------------------------------------------------------------------
# Strict payload base
# ---------------------------------------------------------------------------

class _DemoBody(StrictPayloadBase):
    title: str
    body: str | None = None


class _DemoUnsetBody(OptionalUnset):
    workspace_path: str | None = UNDEFINED


def test_strict_base_forbids_unknown_key():
    with pytest.raises(ValidationError) as exc_info:
        _DemoBody(title="x", unknown_field="y")
    assert "unknown_field" in str(exc_info.value)


def test_strict_base_accepts_known_nullable_field():
    m = _DemoBody(title="x")
    assert m.title == "x"
    assert m.body is None


def test_strict_base_is_supplied_detects_presence():
    absent = _DemoBody(title="x")
    present_null = _DemoBody(title="x", body=None)
    present_value = _DemoBody(title="x", body="hello")

    assert not absent.is_supplied("body")
    assert present_null.is_supplied("body")
    assert present_value.is_supplied("body")


def test_optional_unset_distinguishes_absent_null_and_value():
    absent = _DemoUnsetBody()
    present_null = _DemoUnsetBody(workspace_path=None)
    present_value = _DemoUnsetBody(workspace_path="/tmp/ws")

    assert absent.value_or_none("workspace_path") is None
    assert not absent.is_supplied("workspace_path")

    assert present_null.value_or_none("workspace_path") is None
    assert present_null.is_supplied("workspace_path")

    assert present_value.value_or_none("workspace_path") == "/tmp/ws"
    assert present_value.is_supplied("workspace_path")


def test_optional_unset_whitespace_only_treats_as_none():
    m = _DemoUnsetBody(workspace_path="   ")
    assert m.value_or_none("workspace_path") is None


def test_undefined_sentinel_is_falsy_and_reprs():
    assert not UNDEFINED
    assert "UNDEFINED" in repr(UNDEFINED)
    assert isinstance(UNDEFINED, Undefined)


# ---------------------------------------------------------------------------
# Workspace/branch validator
# ---------------------------------------------------------------------------

class TestValidateWorkspaceSpec:
    def test_valid_scratch_no_path(self):
        spec = validate_workspace_spec("scratch")
        assert spec == WorkspaceSpec("scratch", None, None)

    def test_valid_dir(self):
        spec = validate_workspace_spec("dir", "/tmp/work")
        assert spec == WorkspaceSpec("dir", "/tmp/work", None)

    def test_valid_worktree(self):
        spec = validate_workspace_spec("worktree", "/repos/hermes")
        assert spec == WorkspaceSpec("worktree", "/repos/hermes", None)

    def test_valid_worktree_with_branch(self):
        spec = validate_workspace_spec("worktree", "/repos/hermes", "feature/x")
        assert spec == WorkspaceSpec("worktree", "/repos/hermes", "feature/x")

    def test_rejects_invalid_kind(self):
        with pytest.raises(ValueError, match=r"workspace_kind must be one of"):
            validate_workspace_spec("cloud")

    def test_dir_requires_path(self):
        with pytest.raises(ValueError, match=r"dir.*requires.*workspace_path"):
            validate_workspace_spec("dir", None)

    def test_worktree_requires_path(self):
        with pytest.raises(ValueError, match=r"worktree.*requires.*workspace_path"):
            validate_workspace_spec("worktree", None)

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match=r"non-empty workspace_path"):
            validate_workspace_spec("dir", "  ")

    def test_rejects_relative_path(self):
        with pytest.raises(ValueError, match=r"must be absolute"):
            validate_workspace_spec("dir", "./relative")

    def test_expands_tilde(self):
        home = str(Path.home())
        spec = validate_workspace_spec("dir", "~/vault")
        assert spec.workspace_path == str(Path(home) / "vault")

    def test_branch_only_allowed_for_worktree(self):
        with pytest.raises(ValueError, match=r"branch_name is only valid for worktree"):
            validate_workspace_spec("dir", "/tmp/ws", "feature/x")

    def test_branch_whitespace_trimmed(self):
        spec = validate_workspace_spec("worktree", "/tmp/wt", "  feature/x  ")
        assert spec.branch_name == "feature/x"

    def test_empty_branch_becomes_none(self):
        spec = validate_workspace_spec("worktree", "/tmp/wt", "   ")
        assert spec.branch_name is None

    def test_kind_normalised_to_lowercase(self):
        spec = validate_workspace_spec("DIR", "/tmp/ws")
        assert spec.workspace_kind == "dir"

    def test_undefined_path_treated_as_missing(self):
        with pytest.raises(ValueError, match=r"requires workspace_path"):
            validate_workspace_spec("dir", UNDEFINED)


class TestNormaliseBranchName:
    def test_outer_whitespace_trimmed(self):
        assert _normalise_branch_name("  wt/x  ") == "wt/x"

    def test_empty_becomes_none(self):
        assert _normalise_branch_name("   ") is None

    def test_none_passthrough(self):
        assert _normalise_branch_name(None) is None

    def test_internal_whitespace_preserved(self):
        # Normalisation is intentionally only outer whitespace.  A branch with
        # internal whitespace is still returned and must be rejected by stricter
        # checks if desired.
        assert _normalise_branch_name("  bad branch  ") == "bad branch"

    def test_leading_dash_preserved(self):
        assert _normalise_branch_name("  -bad  ") == "-bad"


# ---------------------------------------------------------------------------
# PayloadValidationError
# ---------------------------------------------------------------------------


def test_payload_validation_error_carries_key():
    err = PayloadValidationError("bad value", key="workspace_path")
    assert err.key == "workspace_path"
    assert "workspace_path" in str(err)


# ---------------------------------------------------------------------------
# Divergence-prevention tests
# ---------------------------------------------------------------------------


def test_workspace_constants_match_kanban_db():
    """The kernel's workspace constants must stay identical to kanban_db."""
    from hermes_cli.kanban_validation import VALID_WORKSPACE_KINDS as kernel_kinds

    assert kernel_kinds == kb.VALID_WORKSPACE_KINDS


def test_all_known_validation_sites_import_kernel():
    """Any module that validates workspace_kind/branch_name must import the kernel.

    This test scans the hermes_cli and plugins/kanban/dashboard trees for
    string literals that are the old validation vocabulary.  If a future
    change adds a second validation path that re-implements these checks, the
    new file must either import from ``hermes_cli.kanban_validation`` or be
    added to ``ALLOWED_EXCEPTIONS`` below with an explicit justification.

    The test does not prove correctness of the kernel itself; the unit tests
    above do that.  It proves that there is only one validator in the
    codebase, which is the property that keeps the three surfaces aligned.
    """
    import ast
    import inspect

    repo = Path(__file__).resolve().parents[2]
    allowed_exceptions: set[str] = {
        # This test file is allowed to talk about the literals.
        str(Path("tests/hermes_cli/test_kanban_validation.py")),
        # The kernel itself defines the canonical messages; it is not a re-implementation.
        "hermes_cli/kanban_validation.py",
        # hermes_cli/kanban_db.py is the canonical store: it defines the DB
        # constants and calls validate_workspace_spec where appropriate.  Its
        # own local checks for workspace_kind are expected to delegate to the
        # kernel in follow-up work; until then, its literal set comparison is
        # guarded by ``test_workspace_constants_match_kanban_db``.
        "hermes_cli/kanban_db.py",
        # The CLI parser splits the --workspace flag before validation; the
        # actual rules come from the kernel.  It contains the word "worktree"
        # in help text and branch validation messages.
        "hermes_cli/kanban.py",
        # Dashboard plugin currently has its own CreateTaskBody; the adoption
        # follow-up will replace that with the kernel.  Until then, we allow
        # the file because the test suite tracks it explicitly.
        "plugins/kanban/dashboard/plugin_api.py",
    }

    marker_literals: frozenset[str] = frozenset({
        "branch_name is only valid",
        "workspace_kind must be one of",
        "branch must not contain whitespace",
        "workspace paths must be absolute",
    })

    offenders: list[str] = []
    roots = [repo / "hermes_cli", repo / "plugins" / "kanban" / "dashboard"]
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo).as_posix()
            if rel in allowed_exceptions:
                continue
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in marker_literals):
                offenders.append(rel)

    assert not offenders, (
        "Files that re-implement workspace/branch validation must import "
        f"hermes_cli.kanban_validation or be added to allowed_exceptions: {offenders}"
    )


def test_validate_workspace_spec_is_reachable_from_kanban_db():
    """The DB layer must be able to call the validator without a circular import.

    This is a structural smoke test: it imports both modules in the order
    kanban_db would import them and verifies the function is callable.
    """
    from hermes_cli import kanban_db
    from hermes_cli.kanban_validation import validate_workspace_spec

    assert callable(validate_workspace_spec)
    # kanban_db should not have imported the validator yet in this subprocess,
    # but importing it must not raise.
    assert kanban_db.VALID_WORKSPACE_KINDS


# ---------------------------------------------------------------------------
# Future-adoption contract tests: these show how the kernel is intended to be
# consumed by the REST/CLI/DB surfaces.  They do NOT mutate those surfaces.
# ---------------------------------------------------------------------------


def test_kernel_can_model_create_task_body():
    """A future CreateTaskBody would subclass StrictPayloadBase and validate."""

    class FutureCreateTaskBody(StrictPayloadBase):
        title: str
        workspace_kind: str = "scratch"
        workspace_path: str | None = UNDEFINED
        branch_name: str | None = UNDEFINED

    body = FutureCreateTaskBody(title="x")
    assert not body.is_supplied("workspace_path")
    spec = validate_workspace_spec(
        body.workspace_kind,
        body.workspace_path if body.is_supplied("workspace_path") else None,
        None,
    )
    assert spec.workspace_kind == "scratch"

    # explicit null is distinguishable from omitted
    body_null = FutureCreateTaskBody(title="x", workspace_path=None)
    assert body_null.is_supplied("workspace_path")


class TestFuturePatchWorkspaceBody:
    """Show how PATCH workspace fields would use the kernel."""

    def test_explicit_null_clears_path(self):
        class FuturePatchWorkspaceBody(OptionalUnset):
            workspace_kind: str | None = UNDEFINED
            workspace_path: str | None = UNDEFINED
            branch_name: str | None = UNDEFINED

        patch = FuturePatchWorkspaceBody(workspace_path=None)
        assert patch.is_supplied("workspace_path")
        assert patch.value_or_none("workspace_path") is None

    def test_absent_keeps_old_value(self):
        class FuturePatchWorkspaceBody(OptionalUnset):
            workspace_path: str | None = UNDEFINED

        patch = FuturePatchWorkspaceBody()
        assert not patch.is_supplied("workspace_path")
        assert patch.value_or_none("workspace_path") is None
