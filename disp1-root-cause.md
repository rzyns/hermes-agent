# DISP-1 incident root-cause note

## Hot loop

A worker-side `kind=dependency` block routes a card to `todo` and does not increment `consecutive_failures`. Before this change, the next dispatcher `recompute_ready()` pass had no memory that the worker had just rejected the same dependency state. The card could therefore be promoted and claimed again while the parent state was byte-for-byte unchanged. The normal failure-limit circuit breaker never accumulated a signal.

The fix persists a fingerprint of board-local parent IDs, statuses, and `completed_at` values when the dependency wait is recorded. Promotion remains parked while that fingerprint is unchanged. Parent completion/status change, link removal/addition, or an explicit operator unblock releases the sticky gate. A same-fingerprint attempt counter emits a deduplicated `dependency_wait_backstop` event at the cap, independently of `consecutive_failures`.

## Why the review fixture reached the live board

Read-only forensic queries of the preserved incident cards found:

- `t_8d4bf6cd`, `t_95104995`, and `t_a87535e0` were all created in the live board at the same second (`created_at=1785145795`).
- Their task rows have `created_by=NULL`, matching direct `kanban_db.create_task()` usage rather than the worker `kanban_create` tool path.
- The first events immediately create/link the exact `never completed archived`, `link-gated`, and `create-gated` matrix.

The process environment of a dispatched Kanban worker intentionally contains `HERMES_KANBAN_DB`, pinned to its live board. `kanban_db.kanban_db_path()` gives that variable highest precedence. Therefore an ad-hoc review script that creates a temporary database but later calls `connect()`/`init_db()` without passing the temporary `db_path` explicitly reconnects to the inherited live board. Merely creating a temp directory or claiming that the matrix is temporary is not an isolation boundary.

This backend change does not alter that platform execution boundary. Follow-up card `t_0645f051` owns the fail-closed verification isolation fix and regression tests. Mutation-capable verification should either run under `scripts/run_tests.sh` (which supplies isolated state) or explicitly remove/override all Kanban board variables and assert the resolved DB path is under the temporary root before the first mutation.

## Evidence preservation

No incident cards were mutated during this work. The forensic query selected only task identity/state and initial event rows from `$HERMES_KANBAN_DB`.
