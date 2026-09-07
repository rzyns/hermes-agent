# Windows validation hardening

## Scope

Close the two validation gaps left by the sidebar request-saturation implementation: four Windows profile-test failures and a Desktop store command terminated by the foreground tool limit. Preserve the sidebar implementation and unrelated local changes. No production profile behavior, runtime configuration, credentials, application installs, services, Git history, or remote instances are changed by this follow-up.

## Profile tests

The profile source and test file matched HEAD before reproduction (`git diff --exit-code HEAD -- hermes_cli/profiles.py tests/hermes_cli/test_profiles.py`). The canonical runner reproduced all four failures: 55 passed, four failed.

- Fresh placeholder and copied backfill tests expected POSIX `0600` from Windows `st_mode`.
- The POSIX wrapper test expected an extensionless shell script on Windows.
- The custom alias listing test expected an extensionless wrapper path on Windows.

Changes in `tests/hermes_cli/test_profiles.py` retain unmarked shared creation/content/isolation checks. Dedicated Linux/macOS items verify real permission bits; each parametrized item carries exactly one native-host marker, allowing the existing CI discovery and selection mechanism to find it. Backfill permission coverage additionally verifies a `0644` source remains unchanged while the destination becomes `0600`. Native Windows coverage verifies the `.bat` path, header, profile target and argument forwarding. Alias listing checks the returned wrapper and the host-specific filename.

No OS detection is mocked. Windows ACL enforcement is neither tested nor claimed. Local Windows execution cannot prove Linux/macOS results. The same six-file combined suite subsequently ran in an isolated patched source archive on Blinky/Linux: 114 passed, zero failed, six platform-specific skips. Native macOS execution remains unverified.

Independent static review passed without findings. Parent reran the profile file: 59 passed, six platform-specific skips. Combined profile/sidebar/off-loop suites: 114 passed, zero failed, six skips across six files.

Evidence:
- `C:/Users/jmdzi/AppData/Local/Temp/profile-tests-red.log`
- `C:/Users/jmdzi/AppData/Local/Temp/profile-tests-green.log`
- `C:/Users/jmdzi/AppData/Local/Temp/profile-tests-review.txt`
- `C:/Users/jmdzi/AppData/Local/Temp/hermes-profile-combined-final.log`

## Desktop store investigation

Ran all `src/store` tests in a background process with two workers, verbose output, a JSON reporter, and a saved log, rather than allowing a foreground tool deadline to terminate the runner:

    npm exec --no -- vitest run src/store --maxWorkers=2 --reporter=verbose --reporter=json --outputFile.json=C:/Users/jmdzi/AppData/Local/Temp/hermes-store-suite.json > C:/Users/jmdzi/AppData/Local/Temp/hermes-store-suite.log 2>&1

The run completed with exit 1. Parsed JSON contains 118 file records and 1,464 assertions: 1,463 passed, one failed. The failure is the first test in `session-unread-tile.test.ts`, which exceeded its existing 15-second test deadline. The remaining two tests in that file passed. JSON suite counters include nested suites; use file records rather than calling those counters file counts.

To separate the failure from the sidebar patch, extracted all tracked `apps/desktop` and `apps/shared` sources and workspace manifests from HEAD `a4f5cd16a7dccd0fbf9652d5a82aeefcbb687cfe` into a temporary directory. Installed dependencies are shared via junctions; no working-tree source was reverted. Running the same file on that baseline reproduced the same first-test timeout, with the other two tests passing. This establishes the timeout also occurs without the sidebar source changes.

Baseline command (from the temporary Desktop directory):

    npm exec --no -- vitest run src/store/session-unread-tile.test.ts --maxWorkers=2 --reporter=verbose

Evidence:
- Full first-run log and JSON: `C:/Users/jmdzi/AppData/Local/Temp/hermes-store-suite.log`, `C:/Users/jmdzi/AppData/Local/Temp/hermes-store-suite.json`
- Isolated HEAD baseline: `C:/Users/jmdzi/AppData/Local/Temp/hermes-store-head-p6pk4957/apps/desktop`
- Baseline result: `C:/Users/jmdzi/AppData/Local/Temp/hermes-unread-head.log`

The fixture's real `createClientSessionState` factory lives in `lib/chat-runtime.ts`, whose unrelated `formatRefValue` import pulls in the transcript renderer and its image/media/session-link UI dependencies. The test now supplies a fail-loud mock for that rendering-only dependency. The factory, pane tree/model, registry, session publication, unread logic, per-test module resets, and all three behavioral assertions remain real and unchanged. No production code or timeout setting changed.

Applied only this test-fixture change to the temporary HEAD source snapshot and reran the same command: exit 0, all three tests passed (first test 5,866 ms; total 10.89 s). This is a second independent check that the correction works without the sidebar patch. The temporary snapshot now contains HEAD production sources plus the corrected test fixture, not a pristine test tree. Result: `C:/Users/jmdzi/AppData/Local/Temp/hermes-unread-head-fixed.log`.

Final full-store verification completed with exit 0. The JSON report and text log agree: 118 files and 1,464 tests passed, with zero failures, pending tests or TODO tests. All 432 nested suites passed; these are not file counts. There were no snapshot failures or updates. Total runner duration was 429.90 seconds, longer than the earlier 420-second foreground limit. The corrected unread-tile file passed all three tests with a reported execution span of 3.762 seconds.

Final evidence: `C:/Users/jmdzi/AppData/Local/Temp/hermes-store-suite-final.json` and `C:/Users/jmdzi/AppData/Local/Temp/hermes-store-suite-final.log`. Individual assertion counts were parsed and checked against declared totals. Desktop typecheck and changed-file ESLint passed; independent review approved the fixture boundary without findings (`C:/Users/jmdzi/AppData/Local/Temp/unread-tile-review.txt`). The implementation subagent timed out before writing its standalone summary, but its successful checks were recorded in its transcript and the parent independently verified the corrected fixture on HEAD production sources.

## Completed local verification gates

1. Preserve the real pane-focus/unread transition assertions and per-test module isolation when correcting the Desktop test fixture.
2. Run the corrected file, related tests, typecheck and changed-file lint.
3. Independently review the new Desktop test change.
4. Finish a fresh full Desktop store run with saved JSON and verify file/assertion counts and process exit status.
5. Update this report and the sidebar plan with final results; run `git diff --check` and confirm unrelated changes remain.

All-suite statements remain limited to the specifically named suites. The separately authorized deployment installed both backend and Desktop changes with rollback material and initial responsiveness checks. See `sidebar-rollout.md` for the executed record and incomplete live acceptance gates. Native Linux execution passed; native macOS and hosted CI results remain unverified. The operator subsequently authorized committing and pushing the reviewed work to `origin/main`.
