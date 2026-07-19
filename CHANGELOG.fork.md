# Fork changelog — rzyns/hermes-agent

This fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
is a personal distribution: upstream `main` plus the features and fixes below, kept in
sync via the `fork-upstream-sync` workflow. This file maps what the fork carries so the
diff against upstream stays reviewable.

## What the fork adds (kept features)

### Features with no upstream equivalent
- **Subscription proxy** (`hermes_cli/proxy/adapters/subscription.py`) — multi-provider
  proxy adapter (now the default `--provider`): routes requests to whichever OAuth
  subscription is active, aggregates `/v1/models` across credentialed providers.
- **Raw model passthrough** (`gateway/raw_provider.py`) — `raw/<provider>/<model>`
  bypass shared by the API server and the subscription proxy.
- **ACP editor filesystem** (`acp_adapter/filesystem.py`) — file tools read/write
  through the editor (Zed live buffers) with disk fallback; `embedded_context` support.
- **Skills hub git fetch** (`tools/skills_hub.py`) — private/source-qualified taps via
  `git`, GitHub tree-URL refs, symlink-safe installs, local-filesystem skill discovery.
- **Kanban operator tooling** — health/backup/repair/maintenance CLI
  (`kanban_health.py`), intake links (`kanban_intake_link*.py` + webhook/dashboard
  surfaces), research-route materialization, artifact manifests, decompose `--dry-run`,
  DB durability hardening, cross-board dependency seam (`kanban_dependencies.py`) with
  the `kanban_cross_deps` plugin, self-heal (creation-time policy + opt-in dispatcher
  repair cards), dashboard endpoints.
- **Fork-safe update flow** — `hermes update-maintenance` (fresh-process maintenance
  for external update drivers), `cli:update` plugin-policy hook, fork-upstream-sync
  workflow, worktree-aware git install detection.
- **Session retitle** — `hermes sessions retitle` backfill with title provenance.
- **Provider warnings** — transports surface `provider_warnings` on assistant messages.
- **Desktop/web** — GitHub-dark colorblind theme, backend port-readiness detection,
  reconnect recovery banner, session source tagging/filtering, kanban dashboard slots.
- **Skills** — `skills/creative/html-artifact` (MIT, adapted from Anthropic's gallery).

### Hardening/fixes to upstream subsystems (upstream-PR candidates)
- MiniMax OAuth source-aware refresh/quarantine/write-through
  (`hermes_cli/auth.py`, `agent/credential_pool.py`).
- Langfuse plugin trace-lifecycle rework (tool-span correlation, session-end close)
  plus the provider-hook payload seam (`provider=` on LLM hooks, `select_tool_schemas`).
- Honcho memory session-key canonicalization; Hindsight reasoning-effort passthrough.
- Session-state hardening: WAL fallback, workspace metadata across compression,
  sqlite corruption guards (`hermes_state.py`).
- Cron per-job profiles + isolation from parallel scripts; verbatim no-agent output.
- MCP stdio serve/watchdog lifecycle fixes; OAuth callback hardening.
- Anthropic OAuth tool-name collision fix (`mcp__hermes__` namespace).
- Moonshot nullable-schema collapse; DeepSeek/etc. provider routing fixes.
- Dashboard login behind reverse-proxy path prefixes; voice mode PipeWire/WSL.
- BlueBubbles Private-API sends/home targets; Mattermost reactions + explicit targets;
  Photon Spectrum space resolution; webhook origin delivery + dynamic routes.
- Profile-plugin isolation for the root CLI (`cli` module shadowing fixes).
- ACP stdout preservation across plugin discovery; oneshot finalizer exit.

## 2026-07-19 cleanup (diff-simplification pass)

Convergence and residue removal against upstream snapshot `ad0ddfb1` (2026-07-18);
roughly −18k diff lines with no intended behavior change to kept features:

- **Merge scars fixed (re-converged on upstream):** managed-Node resolution
  (`find_node_executable`/`with_hermes_node_path`) in whatsapp/web-build/electron
  paths; the `_invalidate_update_cache` + bytecode-clear block in `hermes update`
  (a real regression — stale `.pyc` were never cleared).
- **Deduplicated:** `_run_update_maintenance` now IS the canonical `hermes update`
  maintenance tail (was a ~1,150-line stale copy; `update-maintenance` thereby gains
  external-supervisor handling, managed-node PATH, node-failure messaging);
  subscription proxy imports `_to_openai_base_url` instead of a hand copy.
- **Removed (residue):** langfuse eval-harness scripts + mirror tests (~11k lines,
  see below); governance CLI (~3k lines, see below); kanban JSONL event sidecar
  (disabled prototype with no CLI entry; `kanban_health` backup/repair is the DR
  story); dead provider-warning formatters in `turn_finalizer`; orphaned
  `skills/retrieval-reflex` (needs its GBrain host); committed `evidence/` pytest
  log; stale desktop-parity plan doc; whitespace-only drift in 12 files.
- **BREAKING (fork-only surface):** `hermes kanban link/unlink` lost the flag-based
  cross-board mode (it hard-imported the plugin and wrote edges that were unenforced
  unless the plugin was enabled). Use the plugin CLI instead:
  `hermes kanban-cross-deps add --parent-board A --parent X --child-board B --child Y --kind K`
  (same fields; also `remove/list/status/diagnostics/discover/promote-candidate`).
- **Tests:** `test_safe_update_plugin_cli.py` now skips when the hermes-update-guard
  plugin isn't at its live path (was a guaranteed failure off-box).

### Where removed work lives
- Governance (HGK Surface A/B): branch `hgk-upstream-pr-prep-t_10493012`.
- Langfuse eval pipeline: last commit containing it is `41eb3a5a38` — to keep it on a
  named branch: `git branch archive/langfuse-eval-pipeline 41eb3a5a38 && git push origin archive/langfuse-eval-pipeline`.
- Langfuse lifecycle fix iterations: branch `work/langfuse-upstream-lifecycle-cleanup`
  (superseded by the in-tree plugin rework).

## Known notes
- Upstream `hermes_cli/managed_uv.py` `rebuild_venv()` is a no-op stub (byte-identical
  here); worth an upstream issue.
- Desktop session files (`session-source.ts`, `use-session-actions/*`,
  `electron/main.ts`) sit on upstream's actively-refactored session path — expect
  conflicts on future syncs; prefer converging onto upstream primitives when they land.
