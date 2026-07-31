# ENV-2c run-scoped env consumer enumeration

Scope: every non-test Python source site that reads ``HERMES_KANBAN_*`` or
``HERMES_INFERENCE_*`` environment variables in a code path reachable from a
``delegate_task`` child.

Method: ripgrep for ``os.environ[...]``, ``os.environ.get(...)`` and
``os.getenv(...)`` matching ``HERMES_KANBAN_*`` or ``HERMES_INFERENCE_*``.
Excluded ``tests/`` (test fixtures intentionally manipulate process env).

Result: **zero remaining identity/ownership reads** in non-test source.
Infrastructure path-config reads remain direct (see "Deliberately left direct"
below); they are not worker-identity leaks and DB mutation guards already
fail-closed on delegated children.

Overlay-aware consumers converted in this repair:

| File | Variable(s) | Resolution |
|------|-------------|------------|
| ``agent/kanban_stop.py`` | ``HERMES_KANBAN_PROGRESS``, ``HERMES_KANBAN_STOP_NUDGE``, ``HERMES_KANBAN_TASK`` | ``child_env_lookup``; ``kanban_stop_nudge_enabled`` returns ``False`` under ``is_delegated_child_context()`` |
| ``agent/turn_finalizer.py`` | ``HERMES_KANBAN_TASK``, ``HERMES_KANBAN_RUN_ID`` | ``child_env_lookup``; budget-exhaustion failure recording guarded by ``not is_delegated_child_context()`` |
| ``agent/skill_utils.py`` | ``HERMES_KANBAN_TASK``, ``HERMES_KANBAN_BOARD`` | environment detection returns ``False`` under delegated context; otherwise ``child_env_lookup`` |
| ``agent/conversation_loop.py`` | ``HERMES_KANBAN_TASK`` (log line) | ``child_env_lookup`` (log only reached when nudge fires, but uses overlay for consistency) |
| ``run_agent.py`` | ``HERMES_KANBAN_TASK`` (heartbeat guard) | ``child_env_lookup``; early return already guarded by ``is_delegated_child_context()`` |
| ``model_tools.py`` | ``HERMES_KANBAN_TASK`` | ``child_env_lookup`` (already present from ENV-2b) |
| ``tools/kanban_tools.py`` | ``HERMES_KANBAN_TASK``, ``HERMES_KANBAN_RUN_ID``, ``HERMES_KANBAN_CLAIM_LOCK``, ``HERMES_KANBAN_WORKSPACE`` | ``child_env_lookup``; ownership helpers also reject delegated context; mutators already call ``_reject_delegated_child_mutation`` / ``_assert_not_delegated_child_mutation`` |
| ``tools/send_message_tool.py`` | ``HERMES_KANBAN_TASK`` | ``child_env_lookup`` |
| ``gateway/session_context.py`` | ``HERMES_KANBAN_TASK`` (async-delivery gate) | ``child_env_lookup`` |
| ``hermes_cli/runtime_provider.py`` | ``HERMES_INFERENCE_PROVIDER`` | ``child_env_lookup`` in ``resolve_requested_provider`` and ``canonical_custom_identity`` |
| ``hermes_cli/kanban.py`` | ``HERMES_KANBAN_TASK``, ``HERMES_KANBAN_RUN_ID`` (``_worker_run_id_for``) | ``child_env_lookup`` |
| ``hermes_cli/kanban_db.py`` | ``HERMES_KANBAN_BOARD`` (``get_current_board``) | ``child_env_lookup`` |
| ``tui_gateway/server.py`` | ``HERMES_MODEL``, ``HERMES_INFERENCE_MODEL``, ``HERMES_INFERENCE_PROVIDER`` | ``child_env_lookup`` in ``_resolve_model`` and ``_resolve_startup_runtime`` |

Deliberately left direct (not reachable from a delegated child, or
infrastructure config without identity/ownership implications):

* ``hermes_cli/main.py`` lines 2422-2427, 2497-2500 — chat startup / exit code
  paths that run before any delegated child exists and are intentionally global.
* ``hermes_cli/main.py`` line 3082 — interactive model/provider setup; runs
  before child contexts.
* ``hermes_cli/oneshot.py`` — one-shot CLI entrypoint, no delegated children.
* ``cli.py`` — main CLI process entrypoints (single-query, goal-loop, exit
  codes). Delegated children do not execute the CLI main path.
* ``hermes_cli/doctor.py`` — diagnostic CLI, no delegated children.
* ``hermes_cli/kanban_db.py`` path/config knobs (``HERMES_KANBAN_HOME``,
  ``HERMES_KANBAN_DB``, ``HERMES_KANBAN_WORKSPACES_ROOT``,
  ``HERMES_KANBAN_ATTACHMENTS_ROOT``, ``HERMES_KANBAN_CLAIM_TTL_SECONDS``,
  ``HERMES_KANBAN_CRASH_GRACE_SECONDS``, ``HERMES_KANBAN_BUSY_TIMEOUT_MS``) —
  infrastructure configuration, not worker identity. The overlay scrub set does
  not shadow them (it shadows only the identity/ownership vars). Write paths
  are protected by ``_assert_not_delegated_child_mutation``.
* ``gateway/*`` platform adapters and ``gateway/kanban_watchers.py`` — run in
  their own worker processes / gateway session scope, not inside a
  ``delegate_task`` child thread.
* ``agent/secret_scope.py`` — ``HERMES_INFERENCE_*`` are treated as profile
  launch-time settings, resolved through the active secret scope. The global
  env is only consulted as a legacy fallback when no multiplex scope is
  installed; this is intended credential-resolution behavior, not a run-scoped
  identity leak. Call sites that use these values as runtime model/provider
  selection (``runtime_provider``) now route through ``child_env_lookup`` first.

Verification command used:

```bash
rg 'os\.environ\["HERMES_(KANBAN|INFERENCE)_|os\.environ\.get\("HERMES_(KANBAN|INFERENCE)_|os\.getenv\("HERMES_(KANBAN|INFERENCE)_' -g '!tests' -g '*.py'
```

Remaining matches are either infrastructure path/config reads (listed above)
or pre-child CLI/gateway entry points. Identity/ownership reads in reachable
in-process code paths are all overlay-aware.

## Review-round repair (t_09b67162)

Findings from independent review t_55389010 of commit ea86c4d980:

1. **Absent-at-spawn run-scoped keys could leak.** ``tools/delegate_tool.py``
   only tombstoned a key when it was already present in ``os.environ``. A key
   introduced later (e.g. ``gateway/kanban_watchers.py`` temporarily mutating
   ``HERMES_KANBAN_BOARD``) would fall through to ``os.environ`` in
   ``child_env_lookup``. Fixed by unconditionally setting every
   ``_RUN_SCOPED_ENV_VARS`` key to ``None`` in the child overlay.

2. **Overlay storage was thread-local, not context-local.** The overlay was held
   in ``threading.local``; when ``delegate_tool.py`` copied the context to a
   ``DaemonThreadPoolExecutor`` worker, the overlay did not travel with the
   child. Fixed by moving the overlay to a ``ContextVar
   (``_CHILD_ENV_OVERLAY``) so it propagates through ``contextvars.copy_context``
   and is restored on context exit.

3. **Skill environment cache was context-blind.** ``_ENV_DETECT_CACHE`` keyed
   by env name only, so a parent-warmed ``kanban=True`` result was reused for
   delegated children (and vice versa). Fixed by including the delegated-child
   flag in the cache key.

New regressions added in ``tests/tools/test_delegate_env_leaks.py``:

* ``test_absent_run_scoped_key_introduced_after_child_startup_is_scrubbed``
  introduces ``HERMES_KANBAN_BOARD`` after child startup and asserts the
  child still sees ``None`` while the parent sees the new value.
* ``test_skill_env_cache_is_context_aware`` covers both orderings
  (parent-warm then child, child-warm then parent) and asserts no cross-context
  poisoning.

Repair commit: ``ae18210eff``.
