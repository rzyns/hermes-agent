# Sidebar Request Saturation Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Prevent concurrent all-profile project-tree refreshes from starving ordinary dashboard requests while preserving profile isolation and refresh correctness.

**Architecture:** Coalesce renderer refreshes within their connection/scope epoch; move the expensive synchronous REST tree builder behind an asynchronous single-flight boundary with dedicated bounded execution capacity. Share only in-flight work, not a persistent result cache. Keep response schema and profile-scoped tree construction unchanged.

**Tech Stack:** TypeScript/nanostores/Vitest; Python/FastAPI/AnyIO/pytest.

## Scope and boundaries

Repository: C:/Users/jmdzi/AppData/Local/hermes/hermes-agent. Preserve the pre-existing deleted contributor file and untracked xaa. Initial implementation excluded commits, pushes, running-app replacement, remote deployment, service restarts, credential/config changes, and timeout increases. Deployment and Git publication were subsequently authorized separately; see `sidebar-rollout.md` for current status. This plan fixes the diagnosed request saturation; unrelated Electron deprecations and Windows ps warning are not part of this patch.

## Task 1: Review design before code

Read actual route, renderer refresh callers, connection reset behavior, and tests. Independent reviewer must assess cancellation, context scoping, worker isolation, freshness, cleanup, and integration testing. Incorporate concrete findings before implementation.

Review outcome: approved direction with corrections. Reuse the existing gatewayActivationEpoch() exported by store/gateway.ts, not a new connection framework. Join only flights whose context AND generation remain valid. Separate backend execution capacity from the default AnyIO limiter; locking a synchronous route is insufficient because waiters still occupy workers. Keep coordinator state scoped to an application/event loop, observe detached task failures, and preserve mutation-triggered refresh intent with a trailing read rather than letting a pre-mutation response become authoritative. This patch bounds duplicate work; it does not prove that one cold tree build always finishes within 60 seconds.

Baseline verified: desktop projects.test.ts 40 passed; desktop typecheck passed; profiles sidebar scope tests 8 passed. Test-only Python dependencies live outside the running installation in C:/Users/jmdzi/AppData/Local/Temp/hermes-sidebar-test-venv.

## Task 2: Backend admission and single-flight

Modify hermes_cli/web_routers/profiles.py. Add behavioral regression coverage in tests/hermes_cli/test_profiles_projects_tree_concurrency.py; preserve tests/hermes_cli/test_profiles_sidebar_scope.py.

1. Write an event-synchronized HTTP concurrency regression that blocks tree construction and bursts identical requests. Verify failure on the existing synchronous endpoint.
2. Extract the existing synchronous builder without semantic changes and expose it through an async endpoint. Awaiters must not acquire default AnyIO worker tokens. Bound actual execution separately. Scope in-flight identity to application/event-loop and relevant home/root/parameters; never share across incompatible contexts.
3. Ensure canceling a waiter cannot cancel shared work or admit duplicate running work. Remove completed/failed entries; observe exceptions even if callers disconnect. No durable result cache.
4. Verify failures recover, parameter/home isolation holds, and a synchronous lightweight HTTP route can finish while tree requests remain blocked. Reuse real temporary-profile/SQLite tests to verify original merge and partial-error behavior.

Run from repo root with HERMES_PYTHON=C:/Users/jmdzi/AppData/Local/Temp/hermes-sidebar-test-venv/Scripts/python.exe:
    bash scripts/run_tests.sh tests/hermes_cli/test_profiles_projects_tree_concurrency.py tests/hermes_cli/test_profiles_sidebar_scope.py -j 1

## Task 3: Renderer overlap protection

Modify apps/desktop/src/store/projects.ts and apps/desktop/src/store/projects.test.ts; inspect connection reset owner before choosing identity.

1. Write a deferred-request burst test proving refresh overlap and run it RED.
2. Coalesce requests only within the same valid connection/scope epoch. Preserve mutation refresh intent with a bounded trailing refresh if necessary; do not silently discard structural updates during an in-flight read.
3. Guard success, failure, and loading cleanup against stale contexts, including connection changes and leaving/re-entering all-profiles scope. A new context must not reuse an old server's promise.
4. Verify failed work can be retried and the ordinary single-profile path remains correct.

Run from apps/desktop:
    npm exec --no -- vitest run src/store/projects.test.ts
    npm run typecheck
    npm exec --no -- eslint src/store/projects.ts src/store/projects.test.ts

## Task 4: Review and integrated verification

Independent spec compliance review, then independent correctness/security/quality review. Fix concrete findings with tests and re-review. Re-run both backend suites, related renderer refresh tests, typecheck and changed-file lint. Build local desktop artifacts if feasible without replacing the running packaged app. Run git diff --check and inspect final diff/status. Report exact execution evidence and distinguish validated source from deployment; do not claim Blinky or the running Desktop is fixed before a separately authorized rollout and readback.

## Completion and verification

Local implementation and final independent review are complete. Both blocking review findings were repaired: flight-key construction uses filesystem-free lexical scope inputs, and the additive `after_mutation=true` request waits for any predecessor scan before selecting a fresh shared successor. This preserves mutation freshness even when the original HTTP request timed out while its backend scan survived. Normal reads continue to coalesce without requesting successors.

Final executed verification:
- Canonical Python runner: 55 passed across project-tree concurrency (15), sidebar scope (8), sidebar cache (9), profiles off-loop (18), and config off-loop (5).
- Related Desktop suites: 111 passed across four files selected by projects.test.ts, use-background-sync.test.ts, and connections.test.ts.
- Final projects.test.ts rerun after whitespace cleanup: 50 passed.
- Desktop typecheck passed; changed-file ESLint passed with --max-warnings=0.
- npm run build passed; log: C:/Users/jmdzi/AppData/Local/Temp/hermes-sidebar-final-build.log. Artifacts are in apps/desktop/dist, not installed into release/win-unpacked.
- Independent final combined review approved without blocking security or logic findings. Review record: C:/Users/jmdzi/AppData/Local/Temp/sidebar-final-review.txt.

Follow-up validation closed both earlier gaps: corrected platform-specific test expectations now give 114 passed and six native-host skips across the combined profile/sidebar suites; the full Desktop store run completed with 118 files and 1,464 tests passing, zero failures or pending/TODO tests, in 429.90 seconds. The store suite needed background execution beyond the previous tool deadline and a narrow fixture fix for an unrelated cold renderer import. Both test changes passed independent review. Details and evidence are in `windows-test-hardening.md`.

Verification limits: the combined six-file suite subsequently passed natively on Blinky/Linux (114 passed, zero failed, six platform-specific skips) in an isolated patched source archive. Native macOS execution is still unverified. Existing Vite/build deprecation and bundling warnings remain outside this patch. These are named-suite results, not a claim that the entire repository or live Blinky performance has been validated.

The separately authorized rollout applied the backend patch, restarted the identified service, installed the verified Windows payload, and relaunched Desktop. Initial health/status probes returned HTTP 200 promptly. Authenticated sidebar freshness, mutation checks and the ten-minute observation period remain outstanding. See `sidebar-rollout.md` for evidence and rollback locations. Commit/push to `origin/main` has now been explicitly authorized; the unrelated contributor deletion and untracked xaa remain excluded.
