#!/usr/bin/env python3
"""Plan or repair duplicate OpenAI Codex credential-pool identities.

The command is read-only by default. Mutation requires both ``--apply`` and
``--yes``; every changed auth.json receives a timestamped adjacent backup.
Token material is never emitted, only short SHA-256 fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli.auth import (  # noqa: E402
    _auth_store_lock,
    _decode_jwt_claims,
    _oauth_freshness,
    _oauth_identity,
    _save_auth_store,
)

_PROVIDER = "openai-codex"


@dataclass(frozen=True)
class Removal:
    index: int
    entry_id: str
    label: str
    source: str
    identity: str
    token_fingerprint: str
    survivor_id: str
    survivor_label: str


@dataclass
class FilePlan:
    path: Path
    payload: dict[str, Any]
    source_digest: str
    removals: list[Removal]

    @property
    def changed(self) -> bool:
        return bool(self.removals)


@dataclass
class RepairPlan:
    root: Path
    files: list[FilePlan]
    missing_reference_accounts: list[tuple[str, str]]
    reference_path: Path | None


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _short_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _token_fingerprint(entry: dict[str, Any]) -> str:
    access = entry.get("access_token")
    refresh = entry.get("refresh_token")
    material = f"{access if isinstance(access, str) else ''}\0{refresh if isinstance(refresh, str) else ''}"
    return _short_fingerprint(material)


def _identity_fingerprint(identity: str) -> str:
    return _short_fingerprint(identity)


def _safe_label(entry: dict[str, Any]) -> str:
    for key in ("label", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "<unlabeled>"


def _entry_id(entry: dict[str, Any], index: int) -> str:
    value = entry.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else f"<index:{index}>"


def _codex_rows(payload: dict[str, Any]) -> list[Any]:
    pool = payload.get("credential_pool")
    if not isinstance(pool, dict):
        return []
    rows = pool.get(_PROVIDER)
    return rows if isinstance(rows, list) else []


def _singleton_identity(payload: dict[str, Any]) -> str | None:
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(_PROVIDER)
    if not isinstance(state, dict):
        return None
    tokens = state.get("tokens")
    return _oauth_identity(tokens) if isinstance(tokens, dict) else None


def _account_display(entry: dict[str, Any], identity: str) -> str:
    """Return a non-token account hint for historical missing-account reports."""
    for token in (entry.get("access_token"), entry.get("id_token")):
        claims = _decode_jwt_claims(token)
        if not isinstance(claims, dict):
            continue
        profile = claims.get("https://api.openai.com/profile")
        for email in (
            claims.get("email"),
            profile.get("email") if isinstance(profile, dict) else None,
        ):
            if isinstance(email, str) and email.strip():
                return email.strip()
    label = _safe_label(entry)
    if label != "<unlabeled>":
        return label
    return f"identity_sha256:{_identity_fingerprint(identity)}"


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload, _digest(raw)


def _target_paths(root: Path) -> list[Path]:
    paths = [root / "auth.json"]
    profiles = root / "profiles"
    if profiles.is_dir():
        paths.extend(sorted(path for path in profiles.glob("*/auth.json") if path.is_file()))
    return [path for path in paths if path.is_file()]


def _choose_survivor(
    candidates: list[tuple[int, dict[str, Any]]],
    *,
    prefer_device_code: bool,
) -> tuple[int, dict[str, Any]]:
    if prefer_device_code:
        mirrors = [item for item in candidates if item[1].get("source") == "device_code"]
        if mirrors:
            candidates = mirrors
    return max(candidates, key=lambda item: (_oauth_freshness(item[1]), -item[0]))


def _plan_file(
    path: Path,
    payload: dict[str, Any],
    source_digest: str,
    *,
    is_root: bool,
) -> FilePlan:
    rows = _codex_rows(payload)
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        identity = _oauth_identity(row)
        if identity:
            groups.setdefault(identity, []).append((index, row))

    singleton_identity = _singleton_identity(payload) if is_root else None
    removals: list[Removal] = []
    for identity, candidates in groups.items():
        if len(candidates) < 2:
            continue
        survivor_index, survivor = _choose_survivor(
            candidates,
            prefer_device_code=is_root and identity == singleton_identity,
        )
        survivor_id = _entry_id(survivor, survivor_index)
        for index, row in candidates:
            if index == survivor_index:
                continue
            removals.append(
                Removal(
                    index=index,
                    entry_id=_entry_id(row, index),
                    label=_safe_label(row),
                    source=str(row.get("source") or "<unknown>"),
                    identity=identity,
                    token_fingerprint=_token_fingerprint(row),
                    survivor_id=survivor_id,
                    survivor_label=_safe_label(survivor),
                )
            )
    return FilePlan(path, payload, source_digest, sorted(removals, key=lambda item: item.index))


def _reference_identities(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    payload, _digest_value = _load_payload(path)
    found: dict[str, str] = {}
    for row in _codex_rows(payload):
        if not isinstance(row, dict):
            continue
        identity = _oauth_identity(row)
        if identity:
            found.setdefault(identity, _account_display(row, identity))
    singleton = _singleton_identity(payload)
    if singleton:
        providers = payload.get("providers", {})
        state = providers.get(_PROVIDER, {}) if isinstance(providers, dict) else {}
        tokens = state.get("tokens", {}) if isinstance(state, dict) else {}
        if isinstance(tokens, dict):
            found.setdefault(singleton, _account_display(tokens, singleton))
    return found


def build_plan(root: Path, reference_path: Path | None = None) -> RepairPlan:
    root = root.expanduser().resolve()
    paths = _target_paths(root)
    if not paths:
        raise FileNotFoundError(f"no auth.json files found under {root}")

    loaded = [(path, *_load_payload(path)) for path in paths]
    root_path = root / "auth.json"
    loaded.sort(key=lambda item: (item[0] != root_path, str(item[0])))
    files = [
        _plan_file(
            path,
            payload,
            digest,
            is_root=path == root_path,
        )
        for path, payload, digest in loaded
    ]

    current_root_identities: set[str] = set()
    if files and files[0].path == root_path:
        current_root_identities.update(
            identity
            for row in _codex_rows(files[0].payload)
            if isinstance(row, dict) and (identity := _oauth_identity(row))
        )
        singleton = _singleton_identity(files[0].payload)
        if singleton:
            current_root_identities.add(singleton)

    reference = _reference_identities(reference_path)
    missing = sorted(
        (display, f"identity_sha256:{_identity_fingerprint(identity)}")
        for identity, display in reference.items()
        if identity not in current_root_identities
    )
    return RepairPlan(root, files, missing, reference_path)


def render_plan(plan: RepairPlan) -> str:
    lines = [
        "Codex credential-pool dedupe plan (DRY RUN)" if any(f.changed for f in plan.files) else "Codex credential-pool dedupe plan (no changes)",
        f"Root: {plan.root}",
    ]
    total = 0
    for file_plan in plan.files:
        rel = file_plan.path.relative_to(plan.root)
        lines.append(f"File: {rel}")
        if not file_plan.removals:
            lines.append("  no duplicate decodable identities")
            continue
        rows = _codex_rows(file_plan.payload)
        lines.append(f"  keep {len(rows) - len(file_plan.removals)} row(s); remove {len(file_plan.removals)} duplicate row(s)")
        for removal in file_plan.removals:
            total += 1
            warning = (
                " [misleading label: differs from survivor]"
                if removal.label != removal.survivor_label
                else ""
            )
            lines.append(
                "  REMOVE "
                f"id={removal.entry_id!r} label={removal.label!r} "
                f"source={removal.source!r} "
                f"identity_sha256:{_identity_fingerprint(removal.identity)} "
                f"token_sha256:{removal.token_fingerprint}; "
                f"KEEP id={removal.survivor_id!r} label={removal.survivor_label!r}"
                f"{warning}"
            )
    lines.append(f"Total duplicate rows to remove: {total}")
    if plan.reference_path and plan.reference_path.is_file():
        lines.append(f"Historical reference: {plan.reference_path.name}")
        if plan.missing_reference_accounts:
            lines.append("Accounts present in the historical reference but missing now:")
            for display, fingerprint in plan.missing_reference_accounts:
                lines.append(f"  MISSING {display!r} ({fingerprint}) — re-authentication required")
        else:
            lines.append("No historical reference identities are missing from the root pool.")
    else:
        expected = plan.reference_path.name if plan.reference_path else "auth.json.bak-rename"
        lines.append(f"Historical reference {expected!r} not found; missing-account comparison skipped.")
    lines.append("No token material was printed; fingerprints are SHA-256 prefixes.")
    lines.append("Review this plan before running with --apply --yes.")
    return "\n".join(lines)


def _payload_after_removals(file_plan: FilePlan) -> dict[str, Any]:
    remove_indexes = {item.index for item in file_plan.removals}
    payload = json.loads(json.dumps(file_plan.payload))
    rows = _codex_rows(payload)
    payload["credential_pool"][_PROVIDER] = [
        row for index, row in enumerate(rows) if index not in remove_indexes
    ]
    return payload


def _restore_backup_atomically(backup: Path, target: Path) -> None:
    """Restore an exact backup without exposing a partially copied auth file."""
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.rollback-",
        dir=target.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(backup, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def apply_plan(plan: RepairPlan) -> list[Path]:
    changed = [file_plan for file_plan in plan.files if file_plan.changed]
    if not changed:
        return []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups: list[Path] = []
    with ExitStack() as stack:
        # Match the profile-heal lock order in hermes_cli.auth: profile first,
        # global root second. Root-first here could deadlock with a gateway
        # holding one profile lock while it waits to consolidate into root.
        root_path = plan.root / "auth.json"
        lock_order = sorted(
            changed,
            key=lambda item: (item.path == root_path, str(item.path)),
        )
        for file_plan in lock_order:
            stack.enter_context(_auth_store_lock(target_path=file_plan.path))
        for file_plan in changed:
            current = file_plan.path.read_bytes()
            if _digest(current) != file_plan.source_digest:
                raise RuntimeError(
                    f"{file_plan.path} changed after planning; refusing to apply stale plan"
                )
        backup_by_path: dict[Path, Path] = {}
        for file_plan in changed:
            backup = file_plan.path.with_name(
                f"{file_plan.path.name}.bak-codex-pool-dedupe-{timestamp}"
            )
            if backup.exists():
                raise FileExistsError(f"backup already exists: {backup}")
            shutil.copy2(file_plan.path, backup)
            backups.append(backup)
            backup_by_path[file_plan.path] = backup
        attempted: list[FilePlan] = []
        try:
            for file_plan in changed:
                # Include the current file before writing: a writer that fails
                # after replacing the destination must still be rolled back.
                attempted.append(file_plan)
                _save_auth_store(
                    _payload_after_removals(file_plan),
                    target_path=file_plan.path,
                )
        except Exception as write_error:
            rollback_failures: list[str] = []
            for file_plan in reversed(attempted):
                try:
                    _restore_backup_atomically(
                        backup_by_path[file_plan.path],
                        file_plan.path,
                    )
                except OSError as rollback_error:
                    rollback_failures.append(
                        f"{file_plan.path}: {rollback_error}"
                    )
            if rollback_failures:
                details = "; ".join(rollback_failures)
                raise RuntimeError(
                    f"apply failed and rollback was incomplete: {details}"
                ) from write_error
            raise RuntimeError(
                f"apply failed; restored {len(attempted)} attempted file(s) from backup"
            ) from write_error
    return backups


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan without writing (default)")
    mode.add_argument("--apply", action="store_true", help="apply the reviewed plan")
    parser.add_argument("--yes", action="store_true", help="required confirmation for --apply")
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path.home() / ".hermes",
        help="root Hermes home containing auth.json and profiles/",
    )
    parser.add_argument(
        "--reference-backup",
        type=Path,
        default=None,
        help="historical auth.json used to identify missing accounts (default: <home>/auth.json.bak-rename)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.apply and not args.yes:
        parser.error("--apply requires explicit --yes confirmation")
    if args.yes and not args.apply:
        parser.error("--yes is only valid with --apply")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    reference = args.reference_backup or (args.hermes_home / "auth.json.bak-rename")
    try:
        plan = build_plan(args.hermes_home, reference)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(render_plan(plan))
    if not args.apply:
        return 0
    try:
        backups = apply_plan(plan)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Applied plan to {len(backups)} file(s).")
    for backup in backups:
        print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
