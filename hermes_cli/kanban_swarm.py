"""Kanban Swarm v1: thin swarm topology helpers on top of Kanban.

This module intentionally does not introduce a second scheduler. It writes a
small task graph into the existing Kanban kernel:

    planning root (completed immediately)
        ├─ parallel specialist workers (ready)
        └─ verifier (todo until all workers done)
             └─ synthesizer (todo until verifier done)

The shared blackboard is also deliberately low-tech: structured JSON comments on
the root task. That keeps all state in existing task_comments/task_events rows,
so the dashboard, notifier, slash command, and dispatcher keep working without a
new service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from hermes_cli import kanban_db as kb

BLACKBOARD_PREFIX = "[swarm:blackboard] "
DEFAULT_VERIFIER_SKILLS = ["requesting-code-review"]
DEFAULT_SYNTHESIZER_SKILLS = ["humanizer"]


@dataclass(frozen=True)
class SwarmWorkerSpec:
    """A single parallel worker card in a swarm."""

    profile: str
    title: str
    body: str
    skills: list[str] = field(default_factory=list)
    priority: int = 0
    max_runtime_seconds: Optional[int] = None


@dataclass(frozen=True)
class SelfHealPolicy:
    """Policy for handling operational wiring failures in a swarm.

    Attributes:
        mode: How to react to preflight skill/profile failures.
            - ``fail``: raise before creating any card (default).
            - ``drop``: drop unavailable forced skills; keep task body guidance.
            - ``repair``: create a repair card for broken worker specs; drop
              unavailable optional generated-gate skills so gates still run.
            - ``warn``: create the worker anyway, but log a blackboard warning.
        healer_profile: Profile that receives self-healing repair cards.
        max_repairs_per_swarm: Hard cap on repair cards created per swarm.
        max_repairs_per_task: Hard cap on repair cards per original task.
    """

    mode: Literal["fail", "drop", "repair", "warn"] = "fail"
    healer_profile: str = "default"
    max_repairs_per_swarm: int = 4
    max_repairs_per_task: int = 1

    def __post_init__(self):
        if self.mode not in ("fail", "drop", "repair", "warn"):
            raise ValueError(f"unknown self-heal mode {self.mode!r}")
        if self.max_repairs_per_swarm < 0 or self.max_repairs_per_task < 0:
            raise ValueError("repair caps must be non-negative")


@dataclass(frozen=True)
class SwarmCreated:
    """IDs produced by :func:`create_swarm`."""

    root_id: str
    worker_ids: list[str]
    verifier_id: str
    synthesizer_id: str
    repair_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "worker_ids": list(self.worker_ids),
            "verifier_id": self.verifier_id,
            "synthesizer_id": self.synthesizer_id,
            "repair_ids": list(self.repair_ids),
        }


def _require_text(value: str, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _profile_hermes_home(profile: str, fallback_home: Optional[str] = None) -> str:
    """Resolve the HERMES_HOME the worker subprocess will use for a profile."""

    from hermes_cli.profiles import resolve_profile_env, normalize_profile_name

    try:
        return resolve_profile_env(normalize_profile_name(profile))
    except FileNotFoundError:
        # For tests with fake profiles, fall back to the active HERMES_HOME.
        return fallback_home or os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))


def _external_skill_dirs(hermes_home: str) -> list[Path]:
    """Return configured ``skills.external_dirs`` for a HERMES_HOME."""

    cfg_path = Path(hermes_home) / "config.yaml"
    if not cfg_path.is_file():
        return []
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(hermes_home)
        try:
            cfg = load_config_readonly()
        finally:
            reset_hermes_home_override(token)
        skill_cfg = cfg.get("skills") or {}
        dirs = skill_cfg.get("external_dirs") or skill_cfg.get("external_skill_dirs") or []
        if isinstance(dirs, str):
            dirs = [dirs]
        if not isinstance(dirs, list):
            return []
        roots: list[Path] = []
        for raw in dirs:
            if not raw:
                continue
            expanded = os.path.expanduser(os.path.expandvars(str(raw)))
            path = Path(expanded)
            if not path.is_absolute():
                path = (Path(hermes_home) / path).resolve()
            else:
                path = path.resolve()
            if path.is_dir() and path not in roots:
                roots.append(path)
        return roots
    except Exception:
        return []


def _skill_resolves(skill_name: str, hermes_home: str) -> bool:
    """Check whether a skill identifier resolves under a HERMES_HOME.

    Mirrors the profile-scoped skill discovery the worker CLI will use:
    first local ``<home>/skills/**/<name>/SKILL.md``, then configured
    external skill dirs.  No YAML parsing or prompt building is done,
    so this is cheap enough to run once per worker at swarm creation.
    """

    if not skill_name or not skill_name.strip():
        return False
    name = skill_name.strip()
    roots: list[Path] = [Path(hermes_home) / "skills"]
    roots.extend(_external_skill_dirs(hermes_home))
    for root in roots:
        if not root.is_dir():
            continue
        # Fast path: skill_view-style names may be ``devops/kanban-worker``.
        if (root / name / "SKILL.md").is_file():
            return True
        # Also allow the leaf anywhere under the skill root (bundled layout).
        try:
            for skill_md in root.rglob(f"{name}/SKILL.md"):
                if skill_md.is_file():
                    return True
        except OSError:
            pass
    return False


def preflight_worker_skills(
    spec: SwarmWorkerSpec,
    hermes_home: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Return (available, missing) forced skills for a worker/gate spec.

    By default this resolves the profile-scoped ``HERMES_HOME`` that
    ``_default_spawn`` injects before launching ``hermes -p <profile>``.
    Tests and diagnostic callers may pass ``hermes_home`` to override that
    resolution.  The ``kanban-worker`` skill is treated as a built-in
    dispatcher injection and is excluded from the check.
    """

    fallback_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    home = hermes_home or _profile_hermes_home(spec.profile, fallback_home=fallback_home)
    forced = [s for s in (spec.skills or []) if s and s != "kanban-worker"]
    available = [s for s in forced if _skill_resolves(s, home)]
    missing = [s for s in forced if s not in available]
    return available, missing


def _fail_fast_preflight(
    workers: list[SwarmWorkerSpec],
    hermes_home: Optional[str] = None,
) -> None:
    """Raise before creating any task when any forced skill is missing."""

    failures: list[str] = []
    for spec in workers:
        _, missing = preflight_worker_skills(spec, hermes_home=hermes_home)
        if missing:
            failures.append(
                f"{spec.profile}/{spec.title}: missing {', '.join(missing)}"
            )
    if failures:
        raise ValueError(
            "swarm preflight failed: profile cannot resolve forced skill(s). "
            "Use --self-heal-policy drop|repair|warn, or remove the skills. "
            + "; ".join(failures)
        )


def _append_preflight_manifest(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    author: str,
    entries: list[dict[str, Any]],
) -> None:
    """Append preflight entries without overwriting earlier worker/gate rows."""

    if not entries:
        return
    existing = latest_blackboard(conn, root_id).get("preflight_skill_check")
    manifest = list(existing) if isinstance(existing, list) else []
    manifest.extend(entries)
    post_blackboard_update(
        conn,
        root_id,
        author=author,
        key="preflight_skill_check",
        value=manifest,
    )


def _generated_gate_specs(
    *,
    verifier_assignee: str,
    verifier_title: str,
    synthesizer_assignee: str,
    synthesizer_title: str,
) -> list[SwarmWorkerSpec]:
    return [
        SwarmWorkerSpec(
            profile=verifier_assignee,
            title=verifier_title,
            body="",
            skills=list(DEFAULT_VERIFIER_SKILLS),
        ),
        SwarmWorkerSpec(
            profile=synthesizer_assignee,
            title=synthesizer_title,
            body="",
            skills=list(DEFAULT_SYNTHESIZER_SKILLS),
        ),
    ]


def _resolve_generated_gate_skills(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    gate_specs: list[SwarmWorkerSpec],
    policy: Optional[SelfHealPolicy],
    created_by: str,
) -> tuple[list[str], list[str]]:
    """Resolve forced skills for generated verifier/synthesizer cards.

    Native swarm gate skills are optional methodology helpers: the gate task
    body already contains the required review/synthesis instructions.  Under
    drop/repair policies, remove unavailable generated gate skills so the
    auto-created gate remains dispatchable instead of crashing at CLI startup.
    """

    if policy is None:
        return list(gate_specs[0].skills), list(gate_specs[1].skills)

    resolved: list[list[str]] = []
    manifest: list[dict[str, Any]] = []
    for spec in gate_specs:
        _, missing = preflight_worker_skills(spec)
        if missing:
            if policy is not None and policy.mode in {"drop", "repair"}:
                action = "drop_generated_gate_skill"
                skills = [s for s in spec.skills if s not in missing]
            else:
                action = "warn"
                skills = list(spec.skills)
            manifest.append(
                {
                    "profile": spec.profile,
                    "title": spec.title,
                    "missing_skills": missing,
                    "action": action,
                    "generated_gate": True,
                }
            )
        else:
            skills = list(spec.skills)
        resolved.append(skills)

    _append_preflight_manifest(
        conn,
        root_id,
        author=created_by,
        entries=manifest,
    )
    return resolved[0], resolved[1]


def _apply_preflight_policy(
    conn: sqlite3.Connection,
    root_id: str,
    workers: list[SwarmWorkerSpec],
    policy: SelfHealPolicy,
    created_by: str,
    *,
    tenant: Optional[str] = None,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    priority: int = 0,
    goal: str = "",
    hermes_home: Optional[str] = None,
) -> tuple[list[SwarmWorkerSpec], list[str]]:
    """Validate worker skills and either drop, warn, or create repair cards.

    Returns the (possibly mutated) list of worker specs that should be
    created normally, plus a list of repair/replacement task ids created.
    """

    # Run cheap skill resolution per worker under the profile-scoped
    # HERMES_HOME that the worker process will actually see.
    checks: list[tuple[SwarmWorkerSpec, list[str]]] = []
    for spec in workers:
        _, missing = preflight_worker_skills(spec, hermes_home=hermes_home)
        checks.append((spec, missing))

    any_missing = any(missing for _, missing in checks)
    if not any_missing:
        return workers, []

    # Build a manifest for the blackboard regardless of mode.
    manifest = []
    for spec, missing in checks:
        if missing:
            manifest.append(
                {
                    "profile": spec.profile,
                    "title": spec.title,
                    "missing_skills": missing,
                    "action": policy.mode,
                }
            )

    _append_preflight_manifest(
        conn,
        root_id,
        author=created_by,
        entries=manifest,
    )

    if policy.mode == "warn":
        return workers, []

    repaired: list[str] = []
    kept: list[SwarmWorkerSpec] = []
    for spec, missing in checks:
        if not missing:
            kept.append(spec)
            continue

        if policy.mode == "drop":
            # Keep the worker but remove only the unavailable forced skills.
            # Task body still carries the substantive instructions.
            kept.append(
                SwarmWorkerSpec(
                    profile=spec.profile,
                    title=spec.title,
                    body=spec.body,
                    skills=[s for s in spec.skills if s not in missing],
                    priority=spec.priority,
                    max_runtime_seconds=spec.max_runtime_seconds,
                )
            )
            continue

        if policy.mode == "repair":
            if len(repaired) >= policy.max_repairs_per_swarm:
                # Cap reached: fall back to creating a blocked worker card
                # with no forced skills so the swarm is not silently stunted.
                kept.append(
                    SwarmWorkerSpec(
                        profile=spec.profile,
                        title=spec.title,
                        body=spec.body,
                        skills=[],
                        priority=spec.priority,
                        max_runtime_seconds=spec.max_runtime_seconds,
                    )
                )
                continue

            repair_body = (
                f"Operational repair for swarm worker `{spec.title}` "
                f"(profile `{spec.profile}`).\n\n"
                f"Swarm root: `{root_id}`\n"
                f"Original worker scope: {spec.body or spec.title}\n"
                f"Missing forced skills: {', '.join(missing)}\n"
                f"Swarm goal: {goal}\n\n"
                "Repair actions you may perform:\n"
                "1. Inspect the profile's skill surface and install/sync "
                "the missing skills if that is safe and intended.\n"
                "2. Create a replacement worker card assigned to a profile "
                "that resolves the required skills; link it as a parent "
                "of the verifier.\n"
                "3. Preserve this blocked original card as evidence; do "
                "not archive it until replacements are linked.\n"
                "4. Carry forward every non-authorization from the root blackboard."
            )
            repair_id = kb.create_task(
                conn,
                title=f"Repair: {spec.title}",
                body=repair_body,
                assignee=policy.healer_profile,
                created_by=created_by,
                parents=[root_id],
                tenant=tenant,
                priority=priority + 1,
                workspace_kind=workspace_kind,
                workspace_path=workspace_path,
                skills=["kanban-worker"],
                initial_status="blocked",
            )
            kb.add_comment(
                conn,
                repair_id,
                author=created_by,
                body=(
                    f"Self-healing repair card created because profile "
                    f"`{spec.profile}` cannot resolve forced skills "
                    f"{', '.join(missing)}. Original worker spec was dropped "
                    f"from the swarm graph; this card replaces it."
                ),
            )
            repaired.append(repair_id)
            # The original worker is NOT created; the repair card is the
            # placeholder. Downstream verifier is gated until the repair is
            # completed by the healer profile.
            continue

    if repaired:
        post_blackboard_update(
            conn,
            root_id,
            author=created_by,
            key="self_heal_repairs",
            value=repaired,
        )

    return kept, repaired


def _swarm_context(root_id: str, goal: str) -> str:
    return (
        "\n\n## Swarm protocol\n"
        f"- Swarm root / shared blackboard: `{root_id}`.\n"
        "- Read sibling/parent handoffs from Kanban context before working.\n"
        "- Put machine-readable facts in completion metadata.\n"
        "- Put cross-worker notes on the root task using structured comments.\n"
        f"- Goal: {goal.strip()}\n"
    )


def create_swarm(
    conn: sqlite3.Connection,
    *,
    goal: str,
    workers: Iterable[SwarmWorkerSpec],
    verifier_assignee: str,
    synthesizer_assignee: str,
    root_title: Optional[str] = None,
    verifier_title: str = "Verify swarm outputs",
    synthesizer_title: str = "Synthesize swarm outputs",
    tenant: Optional[str] = None,
    created_by: str = "swarm-orchestrator",
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
    self_heal: Optional[SelfHealPolicy] = None,
) -> SwarmCreated:
    """Create a durable Kanban swarm graph.

    The returned graph is immediately dispatchable: the planning root is marked
    ``done`` with parallel workers are ``ready``, the verifier
    waits for every worker, and the synthesizer waits for the verifier.

    ``self_heal`` enables preflight skill checks. When a worker's or generated
    gate's forced skills cannot be resolved under the profile-scoped
    ``HERMES_HOME`` the spawned worker CLI will see, the policy decides whether
    to fail fast (default), drop unavailable optional skills, create a repair
    card for broken worker specs, or just warn.
    """

    goal = _require_text(goal, "goal")
    verifier_assignee = _require_text(verifier_assignee, "verifier_assignee")
    synthesizer_assignee = _require_text(synthesizer_assignee, "synthesizer_assignee")
    worker_specs = list(workers)
    if not worker_specs:
        raise ValueError("at least one worker is required")
    for i, spec in enumerate(worker_specs, start=1):
        _require_text(spec.profile, f"workers[{i}].profile")
        _require_text(spec.title, f"workers[{i}].title")

    gate_specs = _generated_gate_specs(
        verifier_assignee=verifier_assignee,
        verifier_title=verifier_title,
        synthesizer_assignee=synthesizer_assignee,
        synthesizer_title=synthesizer_title,
    )

    # Fail-fast preflight: when policy is 'fail', validate all forced skills
    # before touching the DB so we never create a half-baked swarm root.
    if self_heal is not None and self_heal.mode == "fail":
        _fail_fast_preflight(worker_specs + gate_specs)

    root = kb.create_task(
        conn,
        title=root_title or f"Swarm: {goal.splitlines()[0][:80]}",
        body=(
            "Kanban Swarm v1 planning/root card. This card is completed "
            "immediately so parallel workers can start while it remains the "
            "shared blackboard and audit anchor.\n\n"
            f"Goal:\n{goal}"
        ),
        assignee=created_by,
        created_by=created_by,
        tenant=tenant,
        priority=priority,
        idempotency_key=idempotency_key,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
    )

    # If idempotency returned an existing non-archived root, do not duplicate the
    # swarm graph. Recover the topology from the root's latest blackboard, if it
    # was created by this helper previously.
    existing = latest_blackboard(conn, root).get("topology")
    if isinstance(existing, dict):
        worker_ids = [str(x) for x in existing.get("worker_ids", []) if x]
        verifier_id = existing.get("verifier_id")
        synthesizer_id = existing.get("synthesizer_id")
        if worker_ids and verifier_id and synthesizer_id:
            return SwarmCreated(
                root_id=root,
                worker_ids=worker_ids,
                verifier_id=str(verifier_id),
                synthesizer_id=str(synthesizer_id),
            )

    kb.complete_task(
        conn,
        root,
        summary="Swarm topology planned; root remains the shared blackboard.",
        metadata={
            "kind": "kanban_swarm_v1",
            "goal": goal,
            "worker_count": len(worker_specs),
        },
    )

    # Self-healing preflight: resolve forced skills under each worker's
    # profile HERMES_HOME before creating the worker cards.
    repair_ids: list[str] = []
    if self_heal is not None:
        worker_specs, repair_ids = _apply_preflight_policy(
            conn,
            root,
            worker_specs,
            self_heal,
            created_by,
            tenant=tenant,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            priority=priority,
            goal=goal,
        )

    context_suffix = _swarm_context(root, goal)
    worker_ids = []
    for spec in worker_specs:
        worker_id = kb.create_task(
            conn,
            title=spec.title,
            body=(spec.body or "") + context_suffix,
            assignee=spec.profile,
            created_by=created_by,
            parents=[root],
            tenant=tenant,
            priority=spec.priority or priority,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            skills=spec.skills or None,
            max_runtime_seconds=spec.max_runtime_seconds,
        )
        worker_ids.append(worker_id)

    # Any self-healing repair cards are evidence-only siblings; they do
    # NOT gate the verifier (the replacement workers produced by the repair
    # policy already do).  Include them in the topology for traceability.
    verifier_parents = list(worker_ids)
    verifier_skills, synthesizer_skills = _resolve_generated_gate_skills(
        conn,
        root,
        gate_specs=gate_specs,
        policy=self_heal,
        created_by=created_by,
    )

    verifier_body = (
        "Review every worker handoff and blackboard update. Gate the swarm: "
        "complete only with metadata {\"gate\": \"pass\"} when evidence is "
        "sufficient; otherwise block with exact missing work."
        + context_suffix
    )
    verifier = kb.create_task(
        conn,
        title=verifier_title,
        body=verifier_body,
        assignee=verifier_assignee,
        created_by=created_by,
        parents=verifier_parents,
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=verifier_skills,
    )

    synthesizer_body = (
        "Synthesize the verified worker outputs into the final deliverable. "
        "Do not start until the verifier has passed the gate."
        + context_suffix
    )
    synthesizer = kb.create_task(
        conn,
        title=synthesizer_title,
        body=synthesizer_body,
        assignee=synthesizer_assignee,
        created_by=created_by,
        parents=[verifier],
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=synthesizer_skills,
    )

    created = SwarmCreated(root, worker_ids, verifier, synthesizer, repair_ids=repair_ids)
    post_blackboard_update(
        conn,
        root,
        author=created_by,
        key="topology",
        value=created.as_dict() | {"goal": goal},
    )
    return created


def post_blackboard_update(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    author: str,
    key: str,
    value: Any,
) -> int:
    """Append one structured update to the swarm root blackboard."""

    _require_text(root_id, "root_id")
    author = _require_text(author, "author")
    key = _require_text(key, "key")
    payload = json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True)
    return kb.add_comment(conn, root_id, author=author, body=BLACKBOARD_PREFIX + payload)


def latest_blackboard(conn: sqlite3.Connection, root_id: str) -> dict[str, Any]:
    """Merge structured blackboard comments on a root card.

    Later comments replace earlier values for the same key. ``_authors`` records
    the author of the winning value for traceability.
    """

    merged: dict[str, Any] = {}
    authors: dict[str, str] = {}
    for comment in kb.list_comments(conn, root_id):
        body = comment.body or ""
        if not body.startswith(BLACKBOARD_PREFIX):
            continue
        try:
            payload = json.loads(body[len(BLACKBOARD_PREFIX):])
        except json.JSONDecodeError:
            continue
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            continue
        merged[key] = payload.get("value")
        authors[key] = comment.author
    if authors:
        merged["_authors"] = authors
    return merged


def parse_worker_arg(raw: str) -> SwarmWorkerSpec:
    """Parse CLI ``--worker profile:title[:skill,skill]`` values."""

    parts = [p.strip() for p in raw.split(":", 2)]
    if len(parts) < 2:
        raise ValueError("worker must be profile:title or profile:title:skill,skill")
    skills: list[str] = []
    if len(parts) == 3 and parts[2]:
        skills = [s.strip() for s in parts[2].split(",") if s.strip()]
    return SwarmWorkerSpec(profile=parts[0], title=parts[1], body=parts[1], skills=skills)
