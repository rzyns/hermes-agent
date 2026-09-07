# Sidebar saturation controlled rollout

## Status and scope

Deployment and restarts were subsequently authorized and completed on 2026-09-07, backend first. The operator also approved the bounded live checks below, then separately authorized committing and pushing this work to `origin/main`. Preserve unrelated local changes and all profile/config/auth/session data. Live acceptance remains incomplete; deployment is not proof of sustained performance.

## Executed rollout record

- Blinky's live router matched the patch baseline and its worktree was clean. The old Desktop closed normally before replacement.
- The canonical Python runner passed in an isolated patched source archive on Blinky/Linux: 114 passed, zero failed, six platform-specific skips across the six related files, in 16.1 seconds. The archive lacked `.git`, producing a pre-compilation diagnostic, but test execution completed with exit 0. Native macOS execution is still unverified.
- Applied only the backend router patch. Readback SHA-256: `d49c65f14eac36641f8f6a6f491f69b71d53d005ea25c303e887d99b9d09bc5c`. Exact remote rollback material: `/home/openclaw/.local/state/hermes-rollouts/sidebar-lx4vt2p0`.
- Restarted only `hermes-dashboard.service`; it reported active/running with new PID 501152. Initial HTTP checks returned 200: health in 0.032 seconds, status in 0.937 seconds. These were direct probes, not authenticated Desktop acceptance checks.
- Installed the staged Desktop payload and verified all 460 files against staging. Retained the old payload at `apps/desktop/release/win-unpacked.before-sidebar-lx4vt2p0`, in addition to the staging rollback copy. Relaunched Desktop as PID 69664 and detected its window. Installed archive SHA-256: `962f3ea0550a10ddfe0d7ad5014fb0a342287690f88d4331f4e902b715c51a04`.
- Still outstanding: authenticated profile/sidebar checks, refresh burst and context switching, disposable project mutations/cleanup, and the ten-minute observation period. No disposable test projects were created during this rollout.

Evidence in the staging directory: `evidence/backend-deploy.log`, `evidence/linux-predeploy-tests.log`, and `desktop-install.json`. PIDs describe the observed rollout, not permanent service identity. No credentials or user data were copied or changed.

## Preparation record (before deployment)

Staging directory: `C:/Users/jmdzi/AppData/Local/Temp/hermes-sidebar-rollout-72kuq0b6`.

- `backend.patch`: only `hermes_cli/web_routers/profiles.py`, based on `a4f5cd16a7dccd0fbf9652d5a82aeefcbb687cfe`.
- `reviewed-source/`: exact relevant production and test files; excludes unrelated contributor deletion and `xaa`.
- `evidence/`: final store JSON/log, Python combined result and independent reviews.
- `manifest.json`: baseline revision and source/patch SHA-256 identities.
- `desktop/win-unpacked/`: successfully built Windows x64 package, not the running `apps/desktop/release/win-unpacked` tree. Static package verification passed; it has not been launched.
- `rollback/desktop-win-unpacked/`: full existing Desktop payload backup; all 460 file hashes matched before/copy/after. This is application code, not user data. Remote backend rollback material must still be captured on Blinky during preflight.
- `package-verification.json`: 382 dist files checked; 380 are byte-identical to build output. Two native dependency package manifests differ only by the packager removing scripts/keywords/bugs metadata. The PE is x64, the renderer includes `after_mutation`, and reviewed source plus the existing installed executable/archive hashes remain unchanged.
- `rollout-review.json`: independent plan review approved with no blockers. Approval covers the plan, not authorization to execute it or proof of live behavior.

Patch application was exercised against a temporary copy of the local HEAD router. `git apply --check` passed and the applied result matched the reviewed source byte-for-byte. This does NOT establish applicability to Blinky; remote drift must be checked before a write.

## Preflight and rollback material

1. Agree on the restart window and ensure no active turns/jobs are interrupted. Determine which dashboard/serve process actually handles the Desktop connection. Discover the real service name, executable, source root and revision from the live target rather than reusing stale paths. Do not read/export environment variables, credentials or auth files.
2. On Blinky, inspect the router file and worktree status, compare its hash/base against the patch, and run an apply check. If drift exists, stop and review a target-specific patch; never overwrite the whole remote checkout or use a forced reset.
3. Before modification, retain the exact target router locally on Blinky outside its live source path, with hash and restrictive access. The local HEAD copy is test material, NOT a verified backup of the remote target. Keep all profile/config/DB files untouched; this change requires no migration.
4. Retain a complete copy of the currently installed Desktop payload (not user data), with verified file hashes. Never pack over the running payload. Verify staged package content against build output and record the new archive/executable hashes. A static package check is not a live application smoke test.
5. Native Linux/macOS profile tests remain outstanding. Run them in the repository's native CI lanes or an explicitly approved isolated environment before claiming cross-platform validation. Do not push merely to trigger CI without authorization.

## Deployment sequence — only in the agreed window

1. Close the old Desktop gracefully so it cannot keep flooding tree requests; do not terminate unrelated Hermes sessions/processes. Pause if unsaved work or active requests make closure unsafe.
2. Apply only the checked backend patch on Blinky and read back the exact target file/hash. Run focused backend tests in an isolated test environment with the canonical runner before restarting. On failure, restore the exact target-file backup before proceeding.
3. Restart only the identified backend service. Verify its actual process/executable/source identity, fresh startup log and responsiveness. Health alone is insufficient: authenticated status/profiles endpoints must respond too. Use the existing authenticated route without exporting tokens.
4. With the old Desktop fully closed, replace only its packaged application directory using the staged payload. Preserve the full old directory for rollback. Do not replace profile data, configuration or authentication files. Read back the installed archive/executable hashes before launch.
5. Start the updated Desktop and verify its actual executable path and packaged source identity. Confirm it connects to the intended backend/profile and that both halves of the change are active.

## Live acceptance checks

Use disposable, operator-approved project records for mutation checks; record IDs and remove only those records afterward. Do not use personal projects as destructive test fixtures.

- Verify ordinary sidebar refresh and switching all-profiles -> specific -> all, including switching away/back while a refresh is outstanding. No stale rows/errors/loading state may overwrite the new context.
- Proposed bounded burst: five normal refresh triggers one second apart, with lightweight endpoint probes during the burst; perform once, not as an endless loop. Stop immediately on a new 60-second API timeout, or if two consecutive lightweight probes exceed five seconds. These trigger counts and stop thresholds require approval along with the restart window.
- Verify project create/rename/delete reconciliation using the disposable records, including a refresh already in flight when the mutation happens. Read back exact results before declaring success.
- Record health/status/profiles timings before and during refresh activity. Proposed initial gate: lightweight requests complete within 5 seconds on this LAN/Tailscale route, with no recurring 60-second API timeouts during a 10-minute observation window. These are proposed operational thresholds, not promises about cold project-tree duration.
- Inspect logs for unhandled errors and sustained repeated tree requests; compare to the diagnosed worker saturation. Do not claim a permanent performance guarantee from one observation window.

## Stop and rollback criteria

Stop on failed patch applicability, failed target tests, wrong running identity, missing authentication, stale profile/mutation results, new unhandled errors, or sustained lightweight endpoint stalls. Do not reset authentication or increase API timeouts to conceal a failure.

If the Desktop update fails, close it and restore the preserved package directory, then verify hashes and launch. If the backend must be rolled back, close the new Desktop first, restore the exact target router backup, restart only the identified backend service, and verify the restored identity and endpoint responses. Restore the prior Desktop as part of a full rollback; do not leave the new client relying on mutation-freshness semantics absent from the old backend. Restore code only, never overwrite current session/config/database state.

## Operator approvals and remaining validation

- Immediate rollout, including backend restart and Desktop close/relaunch, was approved before execution.
- Disposable live mutation checks and the bounded burst/observation thresholds were approved; execution remains outstanding.
- Commit and push to `origin/main` were separately approved. Do not treat a push or queued CI as successful CI execution.

Preparation review and static artifact verification are complete. The build/package command exited 0 with publishing explicitly disabled and output directed to staging. Subsequent deployment is recorded above. Native Linux tests passed in isolation; macOS/hosted CI results and live acceptance remain unverified.

The staging directory is under Temp; preserve it until rollout and rollback observation are complete. The manifest records exact identities because the package version alone does not distinguish this uncommitted local patch from other builds of 0.17.0.
