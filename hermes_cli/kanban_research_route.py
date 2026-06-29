"""Attention Intake → Agent Research route materialization helpers.

This module keeps the safety boundary narrow: it can materialize a durable
Agent Research Intake follow-up card for an Attention Intake source, but the
default and currently supported mode is ``materialize_only``. That creates an
inert blocked/unassigned target plus register provenance; it does not authorize
research execution, content adoption, service changes, or any external action.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root

SOURCE_BOARD = "attention-intake"
TARGET_BOARD = "agent-research-intake"
REGISTER_ROUTE_EVENT = "attention_research_route_materialized"
ROUTE_MATERIALIZATION_MODES = {"materialize_only"}
AUTHORITY_BOUNDARY = (
    "READ-ONLY / PROPOSAL-ONLY; no clone/install/run/import/hook enablement/"
    "skill-memory-config-profile-plugin-cron mutation/provider changes/service "
    "restart/external/public action without separate approval gate."
)


@dataclass(frozen=True)
class RouteMaterializationResult:
    ok: bool
    created: bool
    source_board: str
    source_task: str
    target_board: str
    target_task: str
    target_workspace: str
    materialization_mode: str
    requested_materialization_mode: str
    attention_register_jsonl: str
    target_register_jsonl: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        raise ValueError("url is required")
    return text


def _register_paths(home: Path, board: str) -> tuple[Path, Path]:
    root = home / "artifacts" / board
    return root / "register.md", root / "register.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _write_markdown_register(path: Path, board: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {board} register",
        "",
        "| task | url | status | routed_to | updated |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        task = row.get("task_id") or row.get("source_task") or ""
        url = row.get("url") or row.get("source_url") or ""
        status = row.get("status") or row.get("final_status") or row.get("event") or ""
        routed = row.get("routed_to_task") or ""
        if row.get("routed_to_board") and routed and "/" not in str(routed):
            routed = f"{row['routed_to_board']}/{routed}"
        updated = row.get("route_materialized_at") or row.get("updated_at") or row.get("created_at") or ""
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(v) for v in [task, url, status, routed, updated]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matching_attention_route_row(row: dict[str, Any], *, source_task: str, url: str) -> bool:
    row_url = row.get("url") or row.get("source_url") or row.get("normalized_url")
    if row_url and str(row_url).strip() != url:
        return False
    row_source = row.get("source_task") or row.get("task_id")
    if row_source and row_source != source_task:
        return False
    if row.get("routed_to_task"):
        return bool(row_source == source_task and (not row_url or str(row_url).strip() == url))
    route_packet = row.get("route_packet") if isinstance(row.get("route_packet"), dict) else {}
    requested_board = (
        row.get("routed_to_board")
        or row.get("routed_to_board_requested")
        or route_packet.get("routed_to_board")
        or route_packet.get("routed_to_board_requested")
    )
    return (
        row.get("verdict") == "route_elsewhere"
        or row.get("recommended_action") == "create_research_intake_card"
        or requested_board == TARGET_BOARD
    )


def _route_metadata(
    *,
    now_iso: str,
    mode: str,
    requested_mode: str,
    source_task: str,
    target_task: str,
    workspace: Path,
) -> dict[str, Any]:
    return {
        "materialized_at": now_iso,
        "materialized_by": "kanban-research-route",
        "source_board": SOURCE_BOARD,
        "source_task": source_task,
        "target_board": TARGET_BOARD,
        "target_task": target_task,
        "target_workspace": str(workspace),
        "mode": mode,
        "requested_mode": requested_mode,
        "authority_boundary": AUTHORITY_BOUNDARY,
        "scope": (
            "route materialization only; target blocked/unassigned with sticky "
            "block event; no content/adoption/implementation/live mutation approval"
        ),
    }


def _upsert_row(rows: list[dict[str, Any]], row: dict[str, Any], *, matcher) -> bool:
    for idx, existing in enumerate(rows):
        if matcher(existing):
            rows[idx] = {**existing, **row}
            return False
    rows.append(row)
    return True


def _patch_latest_source_run_metadata(
    conn,
    *,
    source_task: str,
    target_task: str,
    workspace: Path,
    materialization: dict[str, Any],
) -> None:
    run = conn.execute(
        "SELECT id, metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (source_task,),
    ).fetchone()
    if not run:
        return
    try:
        metadata = json.loads(run["metadata"] or "{}")
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update({
        "routed_to_board": TARGET_BOARD,
        "routed_to_task": target_task,
        "created_followup": f"{TARGET_BOARD}/{target_task}",
        "route_target_workspace": str(workspace),
        "route_materialization": materialization,
    })
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps(metadata, sort_keys=True), run["id"]),
        )


def materialize_attention_research_route(
    *,
    source_task: str,
    url: str,
    route_title: str | None = None,
    route_body: str | None = None,
    source_artifact: str | None = None,
    mode: str = "materialize_only",
    hermes_home: Path | None = None,
) -> RouteMaterializationResult:
    """Materialize a blocked Agent Research follow-up for an Attention item.

    The helper is idempotent by ``attention-intake/source_task`` + URL. It
    creates both boards if needed, requires the source task to exist, creates or
    repairs a durable target workspace under
    ``artifacts/agent-research-intake/routed-attention-intake/<source>/<target>``,
    records a sticky block event on the target, and writes JSONL/Markdown
    register provenance for both boards.
    """
    mode = str(mode or "").strip() or "materialize_only"
    if mode not in ROUTE_MATERIALIZATION_MODES:
        raise ValueError(f"unsupported route materialization mode: {mode!r}")
    source_task = str(source_task or "").strip()
    if not source_task.startswith("t_"):
        raise ValueError("source_task must be a kanban task id")
    url = _canonical_url(url)

    home = Path(hermes_home) if hermes_home is not None else get_default_hermes_root()
    kb.create_board(SOURCE_BOARD, name="Attention Intake")
    kb.create_board(TARGET_BOARD, name="Agent Research Intake")

    with kb.connect(board=SOURCE_BOARD) as source_conn:
        source = kb.get_task(source_conn, source_task)
        if source is None:
            raise ValueError(f"source task not found on {SOURCE_BOARD}: {source_task}")

    idempotency_key = f"{SOURCE_BOARD}:{source_task}:{TARGET_BOARD}:{url}"
    base_workspace = home / "artifacts" / TARGET_BOARD / "routed-attention-intake" / source_task
    title = (route_title or f"Research follow-up for {SOURCE_BOARD}/{source_task}").strip()
    body_intro = (route_body or "Read-only/proposal-only follow-up for routed Attention Intake item.").strip()
    created = False

    with kb.connect(board=TARGET_BOARD) as target_conn:
        existing = target_conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        if existing:
            target_task = str(existing["id"])
        else:
            target_task = kb.create_task(
                target_conn,
                title=title,
                body=body_intro,
                assignee=None,
                created_by="kanban-research-route",
                workspace_kind="dir",
                workspace_path=str(base_workspace),
                idempotency_key=idempotency_key,
                initial_status="blocked",
                board=TARGET_BOARD,
            )
            created = True
        workspace = base_workspace / target_task
        workspace.mkdir(parents=True, exist_ok=True)
        target_body = (
            f"{body_intro}\n\n"
            "Source provenance:\n"
            f"- Source board/task: {SOURCE_BOARD}/{source_task}\n"
            f"- Source URL: {url}\n"
            f"- Source assessment artifact: {source_artifact or '-'}\n"
            f"- Durable workspace: {workspace}\n\n"
            "Authority boundary:\n"
            f"- {AUTHORITY_BOUNDARY}\n"
        )
        kb.set_workspace(
            target_conn,
            target_task,
            workspace_kind="dir",
            workspace_path=workspace,
        )
        kb.update_task_body(target_conn, target_task, target_body)
        kb.record_blocked_hold(
            target_conn,
            target_task,
            reason="materialize_only Attention Intake route target; explicit manual hold",
            payload={
                "source_board": SOURCE_BOARD,
                "source_task": source_task,
                "source_url": url,
                "authority_boundary": AUTHORITY_BOUNDARY,
            },
            clear_assignee=True,
        )
        kb.add_comment(
            target_conn,
            target_task,
            "kanban-research-route",
            (
                f"Materialized from `{SOURCE_BOARD}/{source_task}` for {url}. "
                f"Mode `{mode}` keeps this target blocked/unassigned until a separate authorization gate."
            ),
        )
        target_row = kb.get_task(target_conn, target_task)
        if target_row is None:
            raise RuntimeError(f"target task disappeared after creation: {target_task}")
        if target_row.status != "blocked" or target_row.assignee is not None:
            raise RuntimeError("materialize_only target failed inert-state verification")

    now_iso = _now_iso()
    materialization = _route_metadata(
        now_iso=now_iso,
        mode=mode,
        requested_mode=mode,
        source_task=source_task,
        target_task=target_task,
        workspace=workspace,
    )

    att_md, att_jsonl = _register_paths(home, SOURCE_BOARD)
    target_md, target_jsonl = _register_paths(home, TARGET_BOARD)
    att_rows = _load_jsonl(att_jsonl)
    att_row = {
        "event": REGISTER_ROUTE_EVENT,
        "board": SOURCE_BOARD,
        "task_id": source_task,
        "source_board": SOURCE_BOARD,
        "source_task": source_task,
        "url": url,
        "source_url": url,
        "source_artifact": source_artifact,
        "verdict": "route_elsewhere",
        "recommended_action": "create_research_intake_card",
        "routed_to_board_requested": TARGET_BOARD,
        "routed_to_board": TARGET_BOARD,
        "routed_to_task": target_task,
        "route_materialized_at": now_iso,
        "route_materialized_by": "kanban-research-route",
        "route_target_workspace": str(workspace),
        "status": "routed",
        "route_materialization": materialization,
    }
    _upsert_row(
        att_rows,
        att_row,
        matcher=lambda row: _matching_attention_route_row(
            row, source_task=source_task, url=url
        ),
    )
    _write_jsonl(att_jsonl, att_rows)
    _write_markdown_register(att_md, SOURCE_BOARD, att_rows)

    target_rows = _load_jsonl(target_jsonl)
    target_row = {
        "event": REGISTER_ROUTE_EVENT,
        "board": TARGET_BOARD,
        "task_id": target_task,
        "url": url,
        "source_board": SOURCE_BOARD,
        "source_task": target_task,
        "source_ref": f"{SOURCE_BOARD}/{source_task}",
        "upstream_source_board": SOURCE_BOARD,
        "upstream_source_task": source_task,
        "upstream_source_ref": f"{SOURCE_BOARD}/{source_task}",
        "source_tasks": [target_task, source_task],
        "status": "blocked",
        "final_task": target_task,
        "final_status": "blocked",
        "final_decision": (
            "PENDING / route card materialized as blocked manual hold for "
            "read-only proposal-only research"
        ),
        "target_workspace": str(workspace),
        "authority_boundary": AUTHORITY_BOUNDARY,
        "route_materialization": materialization,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    _upsert_row(
        target_rows,
        target_row,
        matcher=lambda row: row.get("task_id") == target_task
        or (
            row.get("source_task") == source_task
            and (row.get("url") or row.get("source_url")) == url
        ),
    )
    _write_jsonl(target_jsonl, target_rows)
    _write_markdown_register(target_md, TARGET_BOARD, target_rows)

    with kb.connect(board=SOURCE_BOARD) as source_conn:
        _patch_latest_source_run_metadata(
            source_conn,
            source_task=source_task,
            target_task=target_task,
            workspace=workspace,
            materialization=materialization,
        )
        kb.add_comment(
            source_conn,
            source_task,
            "kanban-research-route",
            (
                f"Materialized route target `{TARGET_BOARD}/{target_task}` for {url}. "
                f"Mode `{mode}` keeps target blocked/unassigned pending separate authorization."
            ),
        )

    return RouteMaterializationResult(
        ok=True,
        created=created,
        source_board=SOURCE_BOARD,
        source_task=source_task,
        target_board=TARGET_BOARD,
        target_task=target_task,
        target_workspace=str(workspace),
        materialization_mode=mode,
        requested_materialization_mode=mode,
        attention_register_jsonl=str(att_jsonl),
        target_register_jsonl=str(target_jsonl),
    )
