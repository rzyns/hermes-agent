"""Context-local state for delegate_task child execution.

The parent Hermes process may itself be a Kanban dispatcher worker with
HERMES_KANBAN_* variables in process env. delegate_task children run inside the
same Python process, but they are not dispatcher-owned Kanban workers. This
module lets code paths that resolve tool schemas or spawn subprocesses fail
closed for delegated children without mutating global os.environ for the parent.

Cron jobs need the same treatment for the same reason: ``cronjob(action="run")``
executes ``run_job()`` in-process, so a cron agent fired from inside a Kanban
worker would otherwise inherit that worker's dispatcher identity.
``non_dispatcher_owned_context()`` covers both cases.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Mapping, MutableMapping

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_delegated_child_context",
    default=False,
)

# Set for any in-process execution that is NOT the dispatcher-owned worker even
# though the worker's HERMES_KANBAN_* vars are legitimately in os.environ (cron
# jobs fired via the `cronjob` tool).  Kept separate from
# _DELEGATED_CHILD_CONTEXT so the delegate_task-specific behaviour attached to
# that flag (subprocess env scrubbing, its own error strings) is unchanged.
_NON_DISPATCHER_OWNED_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_non_dispatcher_owned_context",
    default=False,
)

DELEGATED_CHILD_ENV_MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"

KANBAN_ENV_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BRANCH",
)

# Run-scoped HERMES_* variables that identify the current worker's own task
# and launch-time inference settings. They must NOT propagate into delegated
# sub-agents: a child should have its own identity and resolve its own model
# rather than inheriting the parent's Kanban task or a forced inference seed.
_RUN_SCOPED_ENV_VARS: frozenset[str] = frozenset(
    {
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BRANCH",
        "HERMES_INFERENCE_MODEL",
        "HERMES_INFERENCE_PROVIDER",
    }
)


# Context-local overlay: per-call environment map that shadows os.environ for
# in-process resolution without mutating the process-global mapping. Using a
# ContextVar means the overlay travels with the delegated-child context when it
# is copied to a worker thread (e.g. by DaemonThreadPoolExecutor).
# A value of ``None`` means the key is explicitly removed (returning ``default``
# rather than falling back to the parent's process env).
_CHILD_ENV_OVERLAY: ContextVar[dict[str, str | None] | None] = ContextVar(
    "hermes_child_env_overlay",
    default=None,
)


@contextmanager
def delegated_child_context(
    session_id: str | None = None,
    *,
    overlay: Mapping[str, str | None] | None = None,
) -> Iterator[None]:
    """Mark child execution and isolate its task-local state.

    ``overlay`` is a per-child environment map that can shadow process env
    without mutating ``os.environ``.  When provided, code that resolves
    run-scoped HERMES_* variables through :func:`child_env_lookup` will see
    the overlay values (or ``default`` for explicitly removed keys) for this
    thread only.

    Child construction calls ``set_current_session_id`` internally, so even a
    context entered without an id must restore the parent's ContextVar.  Child
    execution passes its explicit id and receives it only for this scope.
    """
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    overlay_token = _CHILD_ENV_OVERLAY.set(dict(overlay) if overlay is not None else None)
    try:
        # Import lazily: session_context calls is_delegated_child_context() when
        # deciding whether the compatibility os.environ mirror is safe.
        from gateway.session_context import scoped_current_session_id

        with scoped_current_session_id(session_id):
            yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)
        _CHILD_ENV_OVERLAY.reset(overlay_token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get())


@contextmanager
def non_dispatcher_owned_context() -> Iterator[None]:
    """Mark in-process execution that does NOT own the dispatcher's Kanban task.

    A Kanban worker is a normal CLI agent whose default toolset includes
    ``cronjob``; ``cronjob(action="run")`` runs ``run_job()`` inside the worker's
    own process, where ``HERMES_KANBAN_TASK`` is legitimately set.  Without this
    marker the cron agent is misread as that worker: the kanban toolset is
    force-added, the worker protocol is injected into its system prompt, and
    ``kanban_complete`` defaults ``task_id`` to ``$HERMES_KANBAN_TASK`` — letting
    an unrelated cron job close the worker's task and overwrite real results.

    Scoped via ContextVar rather than by clearing ``os.environ``: the env is
    process-global and shared with the worker's own claim heartbeat, the
    gateway's Kanban watchers, and concurrent cron jobs on the parallel pool, so
    mutating it would starve the worker's claim and race those readers.
    """
    token = _NON_DISPATCHER_OWNED_CONTEXT.set(True)
    try:
        yield
    finally:
        _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


def is_dispatcher_owned_worker_context() -> bool:
    """Return True only when this execution owns the dispatcher's Kanban task.

    The single predicate every ``HERMES_KANBAN_*`` identity gate should use
    before trusting those vars.  False for delegate_task children and for cron
    jobs fired in-process from a worker.
    """
    if _DELEGATED_CHILD_CONTEXT.get():
        return False
    return not _NON_DISPATCHER_OWNED_CONTEXT.get()


def enter_non_dispatcher_owned_context() -> Token[bool]:
    """Token-based form of :func:`non_dispatcher_owned_context`.

    For callers whose scope is a long ``try`` with a matching ``finally`` rather
    than a ``with`` block (``cron.scheduler.run_job``).  Pair with
    :func:`exit_non_dispatcher_owned_context`.
    """
    return _NON_DISPATCHER_OWNED_CONTEXT.set(True)


def exit_non_dispatcher_owned_context(token: Token[bool]) -> None:
    """Restore the flag saved by :func:`enter_non_dispatcher_owned_context`."""
    _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    import os

    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(
        os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    )


def child_env_overlay() -> dict[str, str | None] | None:
    """Return the current context's child env overlay, if any."""
    return _CHILD_ENV_OVERLAY.get()


def child_env_lookup(key: str, default: str | None = None) -> str | None:
    """Resolve a single env var respecting the child env overlay.

    In a delegate_task child context with an active overlay, the overlay is
    authoritative for the keys it contains:

    - A string value is returned directly.
    - ``None`` means the key was explicitly removed; ``default`` is returned
      (this prevents a scrubbed run-scoped var from falling back to the
      parent's process env).

    Outside child contexts the process environment is used directly.
    """
    overlay = child_env_overlay()
    if overlay is not None and is_delegated_child_context():
        if key in overlay:
            value = overlay[key]
            if value is None:
                return default
            return value
    import os

    return os.environ.get(key, default)


def scrub_kanban_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    """Return *env* with dispatcher-only Kanban variables removed."""
    cleaned = dict(env)
    for key in KANBAN_ENV_KEYS:
        cleaned.pop(key, None)
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def scrub_run_scoped_env(
    env: Mapping[str, str],
) -> dict[str, str]:
    """Return *env* with run-scoped HERMES_* vars removed.

    This is the same scrub set used at the in-process delegation boundary so
    that subprocess consumers (terminal, execute_code, ACP, etc.) inherit a
    consistent child identity.
    """
    return {k: v for k, v in env.items() if k not in _RUN_SCOPED_ENV_VARS}


def delegated_child_subprocess_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return an env override only when delegated-child lineage must cross fork.

    Most subprocess call sites historically used ``env=None`` to inherit the
    process environment.  In a ``delegate_task`` child, inheriting as-is leaks
    parent dispatcher ``HERMES_KANBAN_*`` vars while losing the ContextVar in
    the new process.  This helper preserves normal ``env=None`` semantics for
    non-delegated calls, and only materializes a scrubbed env when the lineage
    marker must be propagated across a child-process boundary.
    """
    if not is_delegated_child_process_context():
        return None if env is None else dict(env)

    if env is None:
        import os

        env = os.environ
    base = dict(env)
    # Apply the in-process overlay first, if present, so subprocess consumers
    # inherit the same shadowed values this thread sees.  Drop overlay entries
    # whose value is ``None`` (explicit removal) instead of materialising them.
    overlay = child_env_overlay()
    if overlay is not None:
        for k, v in overlay.items():
            if v is None:
                base.pop(k, None)
            else:
                base[k] = v
    # Then strip run-scoped identity and mark the child lineage.
    base = {k: v for k, v in base.items() if v is not None}
    base = scrub_run_scoped_env(base)
    base[DELEGATED_CHILD_ENV_MARKER] = "1"
    return base
