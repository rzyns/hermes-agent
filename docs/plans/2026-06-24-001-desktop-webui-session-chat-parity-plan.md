---
title: "Port WebUI session and chat affordances to Hermes Desktop"
status: active
date: 2026-06-24
type: implementation-plan
target_repo: hermes-agent
scope: apps/desktop, tui_gateway, hermes_state, hermes_cli/web_server
origin: user request for Desktop/WebUI Sessions + Chat parity
---

# Port WebUI session and chat affordances to Hermes Desktop

## Summary

Bring the useful Hermes WebUI session/chat affordances into Hermes Desktop without copying WebUI implementation quirks. The Desktop app already has strong primitives for several of these features: an `assistant-ui` transcript, sticky user-message containers, a prompt timeline, a floating scroll-to-bottom button, persisted per-session CWD state, workspace grouping, and backend session lineage support. The plan is therefore to make the existing Desktop primitives discoverable and lineage-aware, while using the backend's canonical `SessionDB` metadata instead of title/string-prefix heuristics.

The work should land in dependency order:

1. **Foundation metadata:** expose stable message ids and explicit session-family/branch metadata to Desktop.
2. **Chat anchors/navigation:** add visible assistant-response anchors, robust start/bottom navigation, and offscreen-target handling using the existing thread scroll owner.
3. **Workspace selector:** expose Desktop's already-existing CWD/session workspace control in the composer.
4. **Fork from message:** replace Desktop's current metadata-losing branch path with a backend branch operation that records parent/branch lineage.
5. **Session hierarchy and forks:** render compression continuations and forks as expandable pills/tree rows in the sidebar, backed by explicit lineage metadata.

---

## Evidence from source research

| Area | Current finding | Source |
| --- | --- | --- |
| Desktop transcript rendering | Desktop chat is built around `assistant-ui`; `ThreadMessageList` owns the scroll container and `use-stick-to-bottom` state. It groups user prompts with following assistant turns and uses a render budget with "Show earlier". | `apps/desktop/src/components/assistant-ui/thread.tsx`; `apps/desktop/src/components/assistant-ui/thread-list.tsx` |
| Existing user-message anchors | Sticky user message roots already carry `data-message-id` and `data-role="user"`, so prompt-jump behavior can reuse DOM anchors instead of inventing a parallel transcript. | `apps/desktop/src/components/assistant-ui/thread.tsx` |
| Existing timeline | `ThreadTimeline` already derives user-prompt entries and scrolls to `[data-message-id="..."]`, but only appears as a right-edge rail for sessions with at least 4 user turns. | `apps/desktop/src/components/assistant-ui/thread-timeline.tsx` |
| Existing down-arrow | Desktop already has a floating `ScrollToBottomButton` wired through `requestScrollToBottom()` and `$threadJumpButtonVisible`. | `apps/desktop/src/app/chat/scroll-to-bottom-button.tsx`; `apps/desktop/src/store/thread-scroll.ts` |
| Existing workspace/CWD state | Desktop stores `$currentCwd`, `$currentBranch`, profile-specific CWD localStorage, and `workspaceCwdForNewSession()`. New sessions send `cwd` in `session.create`. | `apps/desktop/src/store/session.ts`; `apps/desktop/src/app/session/hooks/use-session-actions.ts` |
| Existing CWD mutation backend | Active-session CWD changes already use RPC `session.cwd.set`; no-active-session changes call `config.get` with `cwd` to normalize branch/cwd. | `apps/desktop/src/app/session/hooks/use-cwd-actions.ts`; `tui_gateway/server.py` |
| Current Desktop fork bug | Desktop's `branchCurrentSession(messageId)` currently calls `session.create` with only the selected message(s), so it loses `parent_session_id`, `_branched_from`, and true fork lineage. | `apps/desktop/src/app/session/hooks/use-session-actions.ts` |
| Correct backend branch primitive | Gateway `session.branch` writes a new session row with `parent_session_id=<old_key>` and `model_config={"_branched_from": old_key}`. | `tui_gateway/server.py` |
| Canonical lineage model | `SessionDB` already classifies branches via `_branched_from` / legacy `end_reason='branched'`, compression continuations via `parent_session_id + end_reason='compression'`, and projects compression tips with `_lineage_root_id`. | `hermes_state.py` |
| Session list projection | `list_sessions_rich()` hides compression continuations and projects roots forward to their live tip, setting `_lineage_root_id` on projected rows. `model_config` is stripped from list API rows. | `hermes_state.py`; `hermes_cli/web_server.py` |
| Desktop type gap | Desktop `SessionInfo` includes `_lineage_root_id` but not the explicit `parent_session_id` / derived branch fields needed to show collapsed topology. | `apps/desktop/src/types/hermes.ts` |
| Stable message ids are available but dropped | REST session messages come from `SessionDB.get_messages()` which `SELECT *`s the SQLite row, including `messages.id`; Desktop `toChatMessages()` currently replaces that with synthetic `${timestamp}-${index}-${role}` ids. | `hermes_state.py`; `hermes_cli/web_server.py`; `apps/desktop/src/lib/chat-messages.ts` |
| Sidebar grouping boundary | Desktop recents are local-agent sessions; messaging sessions are fetched/rendered separately. Workspace grouping is already `parent repo -> worktree -> sessions`. | `apps/desktop/src/app/chat/sidebar/index.tsx`; `apps/desktop/src/app/chat/sidebar/workspace-groups.ts` |

---

## Product model / terminology

Use these terms consistently in UI code, API fields, and copy:

- **Stored session:** one row in SQLite `sessions`.
- **Segment:** a stored session that is one piece of a logical conversation, usually produced by compression.
- **Logical conversation:** a compression chain rendered as one user-facing conversation entry. The backend already projects this as a root row surfaced under the live tip id plus `_lineage_root_id`.
- **Prior segment:** a previous stored session in the same compression chain.
- **Fork / branch:** an alternate conversation child created from another session/message. A fork must be marked by `model_config._branched_from` or an equivalent derived API field; never infer it from title prefixes.
- **Session family:** the logical conversation plus its prior compression segments and fork descendants.

---

## Requirements

### Chat anchors/navigation

- R1. Every assistant response should expose a discoverable **Response** / **Jump to prompt** affordance that scrolls to the corresponding user prompt for that turn.
- R2. Long chats must expose a reliable way to jump to the start and return to the bottom.
- R3. Anchor navigation must work with Desktop's render budget: if the target prompt is hidden behind "Show earlier", the UI should reveal earlier groups first, then scroll.
- R4. Navigation must not introduce a second scroll owner or fight `use-stick-to-bottom`.
- R5. Keyboard and screen-reader users must get meaningful labels/focus behavior for the new controls.

### Workspace selector

- R6. The composer should show the current workspace/CWD in the normal message-composition area, not only in the file/project rail.
- R7. Users should be able to type/paste a directory path and, where Electron/native support exists, choose a folder from a picker.
- R8. With no active session, the selector stages the CWD for the next session and persists it through the existing workspace-CWD mechanism.
- R9. With an active idle session, the selector calls `session.cwd.set`; with a busy session, it should explain that the CWD can change after the current turn finishes.
- R10. The control should preserve remote-profile-specific CWD storage behavior and display normalized branch information when available.

### Fork conversations

- R11. Desktop should expose fork/branch from both user and assistant messages where the backend can support the selected anchor.
- R12. A fork must preserve backend lineage metadata (`parent_session_id` plus branch marker) so the sidebar can group it under its parent.
- R13. Branching from a message should seed the new conversation with the transcript prefix through that anchor, not only the single selected message.
- R14. The branch operation should refresh/open the new session and preserve profile/CWD/model runtime context.

### Session hierarchy and fork grouping

- R15. The sidebar should render one logical entry for a compression chain, with a pill such as `N prior segments` when hidden segments exist.
- R16. Expanding that pill should show the hidden prior segments in chronological/nested order with enough metadata to inspect/resume/export them.
- R17. Forks should be surfaced as a separate `N forks` pill under the parent logical session.
- R18. Fork grouping must use backend lineage fields, never title prefix/string matching.
- R19. Pagination/search must not make hidden segments or forks unfindable.
- R20. Pins must remain durable across compression by continuing to use the lineage-root key where appropriate.

---

## Non-goals / boundaries

- Do not redesign the entire Desktop sidebar. Add lineage/fork hierarchy as progressive disclosure inside the existing recents/workspace/profile sections.
- Do not merge messaging-platform/cron sessions into this first hierarchy pass. Desktop already renders them as separate sections; include them only after local-agent behavior is stable.
- Do not expose raw `model_config` blobs to Desktop just to discover branch metadata. Add small derived fields/counts instead.
- Do not use title, preview, or string-prefix matching to identify forks.
- Do not add external network calls or live service dependencies for this work.

---

## Architecture principles

1. **Backend lineage is authoritative.** The data model already knows compression and branch edges. Desktop should consume explicit fields from `SessionDB` / API responses.
2. **One scroll owner.** `ThreadMessageList` and `use-stick-to-bottom` own scrolling; anchor controls should request scrolls through a small bridge/helper instead of independently manipulating unrelated ancestors.
3. **Stable ids before advanced behavior.** Message-level forking and robust anchors should first preserve SQLite `messages.id` through REST/gateway responses into Desktop `ChatMessage` metadata.
4. **Progressive disclosure.** Sidebar hierarchy should show compact pills by default and expand only when users ask. Counts should be cheap in list rows; detailed child rows can be lazy-fetched.
5. **Workspace control is exposure, not reinvention.** Desktop already has the CWD state and backend RPC path; the feature is primarily UI placement, validation, and discoverability.
6. **Accessible minimalism.** Small pills/buttons need real labels, 40px-ish hit areas or equivalent, focus-visible states, and no hover-only critical functionality.

---

## Implementation units

### U0. Foundation: stable message ids and session-family metadata

**Goal:** Give Desktop the metadata needed for robust anchors, message-level forks, hierarchy pills, and fork grouping.

**Requirements:** R1-R4, R11-R20

**Likely files:**

- `hermes_state.py`
- `hermes_cli/web_server.py`
- `tui_gateway/server.py`
- `apps/desktop/src/types/hermes.ts`
- `apps/desktop/src/lib/chat-messages.ts`
- `apps/desktop/src/hermes.ts`
- tests under `tests/hermes_state/`, `tests/hermes_cli/`, `apps/desktop/src/lib/*.test.ts`, `apps/desktop/src/hermes.test.ts`

**Approach:**

1. **Expose stable message ids in Desktop types.**
   - Add `id?: number` or `db_id?: number` to `SessionMessage` in `apps/desktop/src/types/hermes.ts` to match the existing REST payload from `SessionDB.get_messages()`.
   - Add `dbMessageId?: number` (or equivalent metadata field) to `ChatMessage` so UI actions can pass a stable anchor without depending on synthetic `assistant-ui` ids.
   - Update `toChatMessages()` to preserve stable ids for stored messages while keeping synthetic ids for live/optimistic/inflight messages.
   - Keep backwards compatibility for live gateway messages, which may not carry DB ids until persisted.

2. **Add derived lineage/family fields to session list rows.**
   - Extend `SessionDB.list_sessions_rich()` / API compaction with cheap derived fields, not raw `model_config`:
     - `parent_session_id?: string | null`
     - `lineage_root_id?: string | null` or continue `_lineage_root_id` but add a public alias
     - `is_branch?: boolean`
     - `branch_parent_session_id?: string | null`
     - `compression_segment_count?: number`
     - `prior_segment_count?: number`
     - `fork_count?: number`
   - Preserve `model_config` stripping in `_compact_session_list_row()`.
   - For projected compression tips, keep the root id available and ensure counts refer to the whole logical conversation.

3. **Add a lazy session-family detail endpoint.**
   - Add a backend endpoint such as `GET /api/sessions/{session_id}/family?profile=...` returning:
     ```ts
     {
       requested_session_id: string
       lineage_root_id: string
       live_tip_session_id: string
       compression_chain: SessionInfo[]
       forks: SessionFamilyNode[]
     }
     ```
   - `compression_chain` should be root -> tip, with each segment carrying id/title/message count/timestamps/cwd/source/profile.
   - `forks` should recursively include branch children and their own compression chain summary.
   - Respect `profile` exactly like existing `get_session_detail` / `get_session_messages`.

4. **Add tests for classification.**
   - Compression child is counted as a prior segment and not as a fork.
   - `_branched_from` child is counted as a fork and remains listable.
   - Delegate/subagent child remains hidden and is not counted as a user-facing fork.
   - Projected compression tip keeps the root id and reports correct prior count.

**Verification:**

- Python unit tests for `SessionDB` lineage classification and family shape.
- FastAPI/web-server tests for list-row derived fields and `/family` profile scoping.
- Desktop unit tests for `toChatMessages()` preserving stable ids.

---

### U1. Chat anchors/navigation in Desktop transcript

**Goal:** Make the WebUI-style "Response", "Start", and bottom navigation affordances visible and robust in Desktop.

**Requirements:** R1-R5

**Dependencies:** U0 stable message metadata is strongly recommended for the final shape, but a DOM-id fallback can be used for purely visual prompt jumps.

**Likely files:**

- `apps/desktop/src/components/assistant-ui/thread.tsx`
- `apps/desktop/src/components/assistant-ui/thread-list.tsx`
- `apps/desktop/src/components/assistant-ui/thread-timeline.tsx`
- `apps/desktop/src/store/thread-scroll.ts`
- `apps/desktop/src/i18n/en.ts`
- `apps/desktop/src/i18n/types.ts`
- new tests under `apps/desktop/src/components/assistant-ui/` or `apps/desktop/src/lib/`

**Approach:**

1. **Create a transcript navigation helper.**
   - Add a small helper that maps each assistant message to the user prompt that owns its turn.
   - Prefer `ChatMessage.dbMessageId` / preserved ids once U0 lands; fall back to current `message.id` for live messages.
   - Reuse the existing turn grouping semantics in `ThreadMessageList`: a user message plus following assistant rows form a turn.

2. **Add a visible assistant response anchor.**
   - Add a compact `Response` / `Jump to prompt` control in `AssistantFooter` or the assistant action bar.
   - Keep it keyboard reachable and give it a clear tooltip/aria-label such as `Jump to prompt for this response`.
   - It should scroll to the sticky user bubble (`data-role="user"`, `data-message-id=...`) for that turn.

3. **Handle targets hidden by render budget.**
   - Extend `ThreadMessageList` with a request bridge, e.g. `requestRevealMessage(messageId)`:
     - if target DOM node exists, scroll to it;
     - if target is in hidden earlier groups, increase `renderBudget` enough to include it, preserve scroll offset using the existing `restoreFromBottomRef`, then scroll after layout;
     - if the target is not in the current transcript, show a non-fatal notification.
   - Avoid calling `scrollIntoView()` on arbitrary ancestors; compute offsets against `[data-slot="aui_thread-viewport"]` as `ThreadTimeline` already does.

4. **Unify Start and bottom controls.**
   - Keep `ScrollToBottomButton` as the bottom/down-arrow implementation; do not add a second competing control.
   - Add a `Start` affordance to the timeline rail or a small transcript navigation cluster:
     - if all earlier groups are rendered, scroll to top;
     - if earlier groups are hidden, reveal earlier groups first, then scroll to top.
   - Consider showing the existing `ThreadTimeline` for fewer than 4 user turns when the user is scrolled up, or add a minimal start/bottom pair independent of timeline tick count.

5. **Polish/accessibility pass.**
   - Ensure icon-only buttons have labels.
   - Use `tabular-nums` for counts if any.
   - Avoid hover-only access to Response/Start controls.
   - Respect `prefers-reduced-motion` for smooth scrolling if the app already has a motion-reduction utility.

**Test scenarios:**

- Assistant response action scrolls to the previous/own user prompt.
- Multiple assistant rows after one user prompt all anchor to the same user prompt.
- Target hidden behind `Show earlier` is revealed before scroll.
- Bottom button remains the only scroll-to-bottom path and still uses `requestScrollToBottom()`.
- Start button reveals hidden history before moving to the top.

---

### U2. Composer workspace selector

**Goal:** Make session CWD/workspace selectable directly from the message composition area.

**Requirements:** R6-R10

**Dependencies:** None; reuses existing CWD state/RPC.

**Likely files:**

- `apps/desktop/src/app/chat/composer/index.tsx`
- `apps/desktop/src/app/chat/composer/types.ts`
- `apps/desktop/src/app/desktop-controller.tsx`
- `apps/desktop/src/app/session/hooks/use-cwd-actions.ts`
- `apps/desktop/src/lib/desktop-fs.ts`
- `apps/desktop/src/store/session.ts`
- `apps/desktop/src/i18n/en.ts`
- `apps/desktop/src/i18n/types.ts`
- tests under `apps/desktop/src/store/session.test.ts` and composer tests

**Approach:**

1. **Add a small `WorkspaceSelector` component near the composer.**
   - Display current CWD as a compact pill/field with basename plus full path in tooltip.
   - Allow expanding to an editable text input.
   - Show branch next to it when `$currentBranch` is non-empty.
   - Show an explicit `No workspace` state when CWD is empty.

2. **Wire to existing actions.**
   - Pass `changeSessionCwd` from `useCwdActions()` down through `desktop-controller.tsx` into `ChatBar` / composer props.
   - With no active session, `changeSessionCwd()` already stages and persists CWD via `setCurrentCwd()` and `config.get`.
   - With an active session, `changeSessionCwd()` already calls `session.cwd.set` and updates runtime info.

3. **Folder picker integration.**
   - Reuse existing Electron bridge/file picker helpers where possible (`onPickFolders`, `desktop-fs`, or a narrow new preload call if needed).
   - In remote mode, keep text entry primary and avoid presenting a local OS picker as if it browsed the remote filesystem.
   - Preserve `sanitizeWorkspaceCwd` behavior from `store/session.ts`.

4. **Busy-session behavior.**
   - `session.cwd.set` returns `session busy` when a turn is running. Surface this as a clear inline/notification message.
   - Do not silently mutate `$currentCwd` for a busy active session unless the backend accepts the change.

5. **Copy and layout.**
   - Keep the control compact; it should not make the composer feel like a settings form.
   - Use a minimum useful hit area and focus-visible styling.
   - Avoid raw path overflow by using `min-w-0`, truncation, and full-path tooltip/copy affordance.

**Test scenarios:**

- No active session: typing a path calls `changeSessionCwd`, updates `$currentCwd`, and seeds the next `session.create` payload.
- Active session: committing a path calls `session.cwd.set` with `session_id` and updates `$currentBranch`.
- Remote profile: remembered CWD remains isolated by remote/profile localStorage key.
- Empty path shows `No workspace` and does not call the backend with an invalid CWD.

---

### U3. Real fork/branch from messages

**Goal:** Replace Desktop's current `session.create`-based pseudo-branch with a real lineage-preserving branch operation, then expose it from both user and assistant messages.

**Requirements:** R11-R14, R18

**Dependencies:** U0 for stable message anchors. U2 is helpful but not required.

**Likely files:**

- `tui_gateway/server.py`
- `hermes_state.py`
- `apps/desktop/src/app/session/hooks/use-session-actions.ts`
- `apps/desktop/src/components/assistant-ui/thread.tsx`
- `apps/desktop/src/lib/chat-messages.ts`
- `apps/desktop/src/i18n/en.ts`
- `apps/desktop/src/i18n/types.ts`
- Python tests for gateway branch behavior
- Desktop tests for branch request payloads/action visibility

**Approach:**

1. **Extend the backend branch RPC instead of bypassing it.**
   - Keep existing `session.branch` behavior for full-session branches.
   - Add optional parameters such as:
     ```json
     {
       "session_id": "live-runtime-id",
       "message_db_id": 123,
       "title": "Branch",
       "mode": "prefix_through_message"
     }
     ```
   - The backend should load/persist the transcript prefix through the selected DB message id, create the new session with `parent_session_id=<source stored id>`, and set a branch marker such as `_branched_from` plus optional `_branched_from_message_id`.
   - Return `session_id`, `stored_session_id`, `parent`, `branched_from_message_id`, `messages`, and `info` so Desktop can open the branch immediately.

2. **Handle live/optimistic messages without DB ids.**
   - If the user branches from an unsaved/live message, either:
     - flush/snapshot current history server-side and branch by runtime index, or
     - disable the action with copy explaining it is available after the message is saved.
   - Prefer server-side runtime index only if it can be made deterministic across display-history prefix and tool rows.

3. **Update Desktop branch action.**
   - Change `branchCurrentSession(messageId?)` to call `session.branch` rather than `session.create`.
   - Pass the stable DB message id when branching from a specific user/assistant message.
   - For the existing assistant footer action, default to branch through that assistant response.
   - Add a user-message action to branch from the prompt itself, likely in the same cluster as restore/edit actions.

4. **Open/refresh behavior.**
   - After branch success, `upsertOptimisticSession()` should include returned lineage fields and profile.
   - Navigate to the new stored session id.
   - Refresh sidebar sessions so the fork pill/count appears under the parent.
   - Preserve CWD/profile/model overrides from the backend response.

5. **Migration/compatibility.**
   - Existing old pseudo-branches created via `session.create` cannot be reliably re-parented without unsafe heuristics. Leave them flat.
   - If old branches have titles like `Branch`, do not infer parentage from titles.

**Test scenarios:**

- `session.branch` without `message_db_id` keeps current full-session branch behavior and writes `_branched_from`.
- `session.branch` with `message_db_id` copies only messages through that anchor and writes `_branched_from_message_id`.
- Branch child appears in `list_sessions_rich()` as listable branch, not hidden child.
- Desktop branch action sends `session.branch`, not `session.create`.
- Branch from user and assistant messages produces the expected payload and opens the returned session.

---

### U4. Sidebar session hierarchy for compression continuations

**Goal:** Show collapsed compression segments under the logical conversation entry, similar to WebUI's prior-turn/piece pill, but backed by explicit metadata.

**Requirements:** R15-R20

**Dependencies:** U0 family metadata.

**Likely files:**

- `apps/desktop/src/types/hermes.ts`
- `apps/desktop/src/hermes.ts`
- `apps/desktop/src/store/session.ts`
- `apps/desktop/src/lib/session-ids.ts`
- `apps/desktop/src/lib/session-search.ts`
- `apps/desktop/src/app/chat/sidebar/index.tsx`
- `apps/desktop/src/app/chat/sidebar/session-row.tsx`
- new `apps/desktop/src/app/chat/sidebar/session-family.ts` or similar
- tests under `apps/desktop/src/store/`, `apps/desktop/src/lib/`, and sidebar component tests if available

**Approach:**

1. **Build a frontend row model.**
   - Add a pure function that takes list rows plus optional loaded family details and returns renderable entries:
     - `logical-session-row`
     - `prior-segments-pill`
     - expanded `prior-segment-row[]`
   - Keep this model independent of workspace/profile grouping so it can be applied inside existing sections.

2. **Show counts cheaply.**
   - If `prior_segment_count > 0`, render a pill such as `3 prior segments`.
   - The pill should be a real button with `aria-expanded` and an accessible label.
   - Use tabular numbers to avoid layout shift.

3. **Lazy-load details on expand.**
   - On pill click, call `/api/sessions/{id}/family` with the row's profile.
   - Cache family details by `(profile, lineageRootOrTipId)` in a lightweight store.
   - Render segment rows with title/preview/message count/time and actions that make sense:
     - inspect messages / resume exact segment if supported;
     - copy id;
     - export if existing export path supports it.

4. **Preserve pin and active semantics.**
   - Continue using `sessionPinId(session)` / `_lineage_root_id` for pins.
   - Highlight selected/active based on the routed/live tip id but show root/segment ids in expanded details.
   - Do not duplicate old and new compression tips in the flat row list.

5. **Search behavior.**
   - Extend `sessionMatchesSearch()` or the row model so search can match hidden segment ids/titles/previews when family details are loaded.
   - For not-yet-loaded hidden children, rely on backend search/id resolution; when a hidden segment matches, load/expand its family and reveal it.

6. **Pagination caveat.**
   - Frontend-only grouping from the current page is insufficient because parents/children can be on different pages. Use backend count/detail fields for correctness.
   - Keep `Load more` totals based on backend `exclude_children` semantics so paging remains stable.

**Test scenarios:**

- A projected compression chain renders one top-level row with `N prior segments`.
- Expanding the pill fetches family details and renders prior segments in correct order.
- Pins keyed by lineage root still match the live tip row.
- Search by root id or tip id finds the logical row.
- Workspace grouping still nests sessions under the same workspace parent/worktree after hierarchy is applied.

---

### U5. Sidebar fork grouping and minimal tree/DAG representation

**Goal:** Collapse branch/fork children under the parent logical session and show them as a separate fork pill/tree, without relying on flaky string matching.

**Requirements:** R17-R20

**Dependencies:** U0 and U3.

**Likely files:**

- same Desktop sidebar files as U4
- `hermes_state.py` family endpoint/query helpers
- `apps/desktop/src/lib/session-search.ts`
- tests for branch family rows

**Approach:**

1. **Use separate pills for separate relationship types.**
   - Compression continuation pill: `N prior segments`.
   - Fork pill: `M forks`.
   - Do not merge them into one ambiguous count.

2. **Render a minimal tree first.**
   - Parent logical session row
     - prior segments (if expanded)
     - forks (if expanded)
       - fork logical session row
       - fork's own prior segments/forks if available
   - Internally treat this as a graph/family model so branch-of-branch and compressed branches are representable.
   - UI can remain tree-shaped initially; DAG visualization can be deferred.

3. **Grouping rules.**
   - A branch child belongs under the family whose stored id matches `_branched_from` / `branch_parent_session_id`.
   - If the branch parent has since compressed, group under the parent's logical row using lineage-root/tip resolution.
   - A branch may itself have a compression chain; render it as one fork row with its own prior-segments pill.

4. **Avoid pagination artifacts.**
   - The list endpoint should include `fork_count` even if fork rows are not in the current page.
   - Expanded forks should come from `/family`, not from whatever rows happen to be in `$sessions`.

5. **Search/open behavior.**
   - Searching for a fork id should reveal/open the fork row even if collapsed under a parent.
   - Selecting a fork should route to the fork's live tip, not the parent.
   - Parent selection should not automatically select fork children.

**Test scenarios:**

- Branch created by `session.branch` appears under parent `M forks`.
- Old flat/pseudo-branch sessions without branch metadata remain flat.
- Branch of branch nests under the intermediate branch.
- Compressed branch shows as one fork row with its own prior-segments count.
- Search by fork id expands the relevant parent family.

---

## Suggested rollout order

1. **Metadata/read-only foundation (U0).** Add derived fields, family endpoint, stable message id preservation, and tests. No UI behavior changes yet except type compatibility.
2. **Safe UI exposure (U2 + part of U1).** Add workspace selector and visible Start/bottom/Response controls using current DOM anchors; do not enable message-level fork until stable ids are wired.
3. **Real branch RPC (U3).** Switch Desktop branch actions from `session.create` to `session.branch`; add message-anchor branch once backend prefix branching is tested.
4. **Compression hierarchy (U4).** Add prior-segments pill and lazy details under logical session rows.
5. **Fork hierarchy (U5).** Add fork pill/tree and search/reveal behavior.
6. **Polish pass.** Keyboard, reduced motion, copy, hit area, visual density, and long-session performance checks.

This order keeps each tranche independently shippable and prevents the sidebar tree from depending on unreliable branch data.

---

## Validation plan

### Python/backend

Run focused tests first, then broader relevant suites:

```bash
pytest tests/hermes_state/test_resolve_resume_session_id.py
pytest tests/hermes_state/test_session_archiving.py
pytest tests/hermes_cli/test_web_server.py
pytest tests/hermes_cli/test_web_server_profile_unification.py
```

Add new tests for:

- `SessionDB` branch/compression/delegate classification and family endpoint shape.
- `GET /api/sessions` derived lineage/count fields.
- `GET /api/sessions/{id}/family` profile scoping and 404 behavior.
- `session.branch` full-session and message-prefix modes.

### Desktop TypeScript/UI

Use workspace scripts from `apps/desktop/package.json`:

```bash
npm run typecheck --workspace apps/desktop
npm run lint --workspace apps/desktop
npm run test:ui --workspace apps/desktop
```

Run focused Vitest targets while iterating, especially:

```bash
npm run test:ui --workspace apps/desktop -- src/lib/chat-messages.test.ts
npm run test:ui --workspace apps/desktop -- src/store/session.test.ts
npm run test:ui --workspace apps/desktop -- src/lib/session-ids.test.ts
npm run test:ui --workspace apps/desktop -- src/hermes.test.ts
```

Add new tests for:

- message-id preservation in `toChatMessages()`;
- assistant response anchor mapping;
- hidden-target reveal behavior;
- workspace selector state transitions;
- branch action RPC payloads;
- sidebar family row modeling, pin preservation, and search matching.

### Manual/local smoke

After automated coverage:

1. Start Desktop dev renderer via `npm run dev:renderer --workspace apps/desktop` or full `npm run dev --workspace apps/desktop` if Electron smoke is needed.
2. Open a long stored session and verify:
   - Response button jumps to owning prompt;
   - Start jumps to the top, including hidden earlier groups;
   - bottom/down arrow still returns to the latest content;
   - timeline still works.
3. Start a new chat, set workspace in composer, send first message, and confirm backend runtime info reports the selected CWD/branch.
4. Branch from a user prompt and assistant response; confirm session list/family endpoint reports `parent_session_id` and branch marker/count.
5. Trigger or fixture a compression chain; confirm one row with `N prior segments` and expandable details.

---

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Anchor scrolling fights `use-stick-to-bottom` and causes jumpy layout. | Route all scroll requests through `ThreadMessageList`/`thread-scroll` bridge; compute offsets against the actual viewport. |
| Render-budget hidden targets make Response/Start feel broken. | Add reveal-before-scroll behavior and tests for hidden groups. |
| Message ids differ between live, stored, and resumed transcripts. | Preserve SQLite ids when available, keep synthetic fallback for live messages, and disable/adjust message-level fork until a stable anchor exists. |
| Exposing raw `model_config` leaks internal/heavy config to high-fanout list APIs. | Add derived booleans/counts and keep `_compact_session_list_row()` stripping raw blobs. |
| Sidebar grouping duplicates rows or breaks pins after compression. | Continue using `sessionPinId()` / lineage-root semantics; test merge/pin cases. |
| Frontend-only grouping fails across pagination. | Use backend count/detail fields and lazy family endpoint instead of relying on the current page. |
| Old Desktop pseudo-branches cannot be grouped. | Leave unmarked historical sessions flat; do not infer parentage from titles. |
| Workspace selector is confused in remote mode. | Text entry remains canonical; only show local folder picker when it browses the same filesystem the backend uses. |

---

## Open questions

1. **Labeling:** Should the assistant-message anchor be literally labeled `Response` to match WebUI, or should Desktop use clearer copy like `Prompt` / `Jump to prompt` with `Response` as tooltip/help text? The implementation can support either; the plan recommends accessible copy `Jump to prompt for this response`.
2. **Exact branch anchor semantics:** For a user-message fork, should the new branch include the selected user prompt and wait for a new assistant response, or include the assistant response that followed it if one exists? The plan recommends `prefix_through_message` with explicit user vs assistant anchor behavior, then UI copy that makes the selected boundary clear.
3. **Segment resume affordance:** Expanded prior compression segments can be inspectable, but normal resume should route to the live tip. If users need exact historical segment resume, add it as an explicit advanced action so it does not surprise users.
4. **DAG visualization:** The first UI should be a minimal tree in the sidebar. A fuller DAG/minimap should wait until branch/family data is correct and users have tried the tree.

---

## Definition of done

- Desktop composer exposes a reliable workspace selector backed by existing CWD/session RPC state.
- Assistant responses have a discoverable anchor action that jumps to the corresponding prompt in long sessions.
- Start and bottom navigation are reliable and do not conflict with Desktop's scroll owner.
- Desktop branch/fork actions use backend lineage-preserving branch semantics rather than `session.create` pseudo-branches.
- Session sidebar rows show expandable prior compression segments and forks using backend metadata, not string matching.
- Hidden/collapsed sessions remain searchable/openable.
- Automated backend and Desktop tests cover the new contracts and critical UI state transitions.
