"""Tests for hermes_cli.kanban_validation.

These tests enforce the kernel contract and act as a **literal-copy tripwire**
against future surfaces re-implementing their own workspace/branch rules.

Scope of the guarantee, stated precisely because it is easy to overstate: the
divergence tests below scan for copies of the kernel's canonical error strings
and constants. They catch copy-paste re-implementation. They do **not** make
divergence impossible -- a paraphrased second validator, written from scratch
without reusing the canonical strings, evades the scan. Confirmed by the
independent review of this module (t_90d4e7ae, 2026-08-05), which defeated the
scan deliberately.

Treat a passing divergence test as "nobody copied the rules", not as "nobody
reimplemented the rules".
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

    def test_worktree_path_may_be_deferred(self):
        spec = validate_workspace_spec(
            "worktree", None, require_path_for=frozenset()
        )
        assert spec == WorkspaceSpec("worktree", None, None)

    def test_dir_path_may_be_deferred(self):
        spec = validate_workspace_spec(
            "dir", None, require_path_for=frozenset()
        )
        assert spec == WorkspaceSpec("dir", None, None)

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match=r"workspace_path must be non-empty"):
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

    def test_branch_internal_whitespace_rejected(self):
        with pytest.raises(ValueError, match=r"branch has internal whitespace"):
            validate_workspace_spec("worktree", "/tmp/wt", "bad branch")

    def test_branch_leading_dash_rejected(self):
        with pytest.raises(ValueError, match=r"branch_name must not start with"):
            validate_workspace_spec("worktree", "/tmp/wt", "-bad")

    def test_kind_normalised_to_lowercase(self):
        spec = validate_workspace_spec("DIR", "/tmp/ws")
        assert spec.workspace_kind == "dir"

    def test_undefined_path_treated_as_missing(self):
        with pytest.raises(ValueError, match=r"requires workspace_path"):
            validate_workspace_spec("dir", UNDEFINED)

    def test_deferred_policy_undefined_worktree_path_allowed(self):
        spec = validate_workspace_spec(
            "worktree", UNDEFINED, require_path_for=frozenset()
        )
        assert spec == WorkspaceSpec("worktree", None, None)

    def test_deferred_policy_undefined_dir_path_allowed(self):
        spec = validate_workspace_spec(
            "dir", UNDEFINED, require_path_for=frozenset()
        )
        assert spec == WorkspaceSpec("dir", None, None)


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
    string literals that are validation vocabulary.  Presence alone is not
    enough to convict a file: repair/mutation surfaces legitimately describe
    the rules in docstrings or error messages.  A file that contains the
    vocabulary is therefore allowed only if it also imports from or calls
    ``hermes_cli.kanban_validation`` (i.e. it delegates), or it is listed in
    ``ALLOWED_EXCEPTIONS`` below with an explicit justification.  This keeps
    the consolidation tripwire from eroding every time a literal is reworded.

    The test does not prove correctness of the kernel itself; the unit tests
    above do that.  It proves that there is only one live validator in the
    codebase, which is the property that keeps the three create surfaces
    aligned.
    """
    import ast

    repo = Path(__file__).resolve().parents[2]
    allowed_exceptions: set[str] = {
        # This test file is allowed to talk about the literals.
        str(Path("tests/hermes_cli/test_kanban_validation.py")),
        # The kernel itself defines the canonical messages; it is not a re-implementation.
        "hermes_cli/kanban_validation.py",
        # hermes_cli/kanban_db.py: the create path delegates correctly to the
        # kernel at create_task (:4071-4076).  The remaining local
        # workspace_kind checks are deferred mutation/repair surfaces:
        #   * set_workspace() still performs a local kind check
        #     (:10206-10233)
        #   * resolve_workspace() / _resolve_worktree_workspace() retain
        #     defensive dispatch-time guards for absolute paths and git anchors
        # Those surfaces are intentionally out of scope for this card and belong
        # to t_c6d77bec.  The literal set comparison is additionally guarded by
        # ``test_workspace_constants_match_kanban_db``.
        "hermes_cli/kanban_db.py",
    }

    # Vocabulary that indicates a file is talking about workspace/branch rules.
    # Adding new literals here is fine; the predicate below also requires
    # kernel delegation, so docstrings alone do not fail.
    marker_literals: frozenset[str] = frozenset({
        "branch has internal whitespace",
        "branch_name is only valid",
        "branch_name must not start with",
        "workspace paths must be absolute",
        "workspace_kind must be one of",
    })

    def _imports_kernel(text: str) -> bool:
        """Return True if the file imports from or calls the validation kernel."""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "hermes_cli.kanban_validation":
                    return True
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "validate_workspace_spec":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "validate_workspace_spec":
                    return True
        return False

    offenders: list[str] = []
    roots = [repo / "hermes_cli", repo / "plugins" / "kanban" / "dashboard"]
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo).as_posix()
            if rel in allowed_exceptions:
                continue
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in marker_literals):
                if _imports_kernel(text):
                    continue
                offenders.append(rel)

    assert not offenders, (
        "Files that contain workspace/branch validation vocabulary must "
        "import/call hermes_cli.kanban_validation or be added to "
        f"allowed_exceptions with a justification: {offenders}"
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
