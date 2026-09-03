"""Tests for the dry-run-first Codex credential-pool repair script."""

from __future__ import annotations

import base64
import contextlib
import json
from pathlib import Path

import pytest

from scripts import codex_pool_dedupe as dedupe


def _jwt(account_id: str, *, email: str | None = None, exp: int = 0) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload: dict = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
        },
        "exp": exp,
    }
    if email:
        payload["https://api.openai.com/profile"] = {"email": email}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"{header}.{encoded}.sig"


def _row(
    entry_id: str,
    label: str,
    source: str,
    account_id: str | None,
    refresh: str,
    *,
    exp: int = 0,
    email: str | None = None,
) -> dict:
    return {
        "id": entry_id,
        "label": label,
        "source": source,
        "auth_type": "oauth",
        "access_token": _jwt(account_id, email=email, exp=exp) if account_id else "opaque-access",
        "refresh_token": refresh,
    }


def _payload(singleton: dict, rows: list[dict]) -> dict:
    return {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": singleton["access_token"],
                    "refresh_token": singleton["refresh_token"],
                }
            },
            "anthropic": {"api_key": "preserve-unrelated-provider"},
        },
        "credential_pool": {
            "openai-codex": rows,
            "anthropic": [{"id": "preserve-unrelated-row"}],
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_dry_run_reports_safe_plan_without_writing_or_printing_tokens(
    tmp_path, capsys
):
    root = tmp_path / ".hermes"
    device = _row("device", "primary", "device_code", "account-A", "secret-refresh-A", exp=10)
    duplicate = _row(
        "manual-duplicate",
        "former-account-label",
        "manual:device_code",
        "account-A",
        "secret-refresh-duplicate",
        exp=20,
    )
    missing = _row(
        "missing",
        "missing-account",
        "manual:device_code",
        "account-B",
        "secret-refresh-B",
        email="missing@example.com",
    )
    auth_path = root / "auth.json"
    _write(auth_path, _payload(device, [device, duplicate]))
    original = auth_path.read_bytes()
    _write(root / "auth.json.bak-rename", _payload(device, [device, missing]))

    assert dedupe.main(["--hermes-home", str(root)]) == 0

    output = capsys.readouterr().out
    assert auth_path.read_bytes() == original
    assert "REMOVE id='manual-duplicate'" in output
    assert "KEEP id='device'" in output
    assert "misleading label" in output
    assert "missing@example.com" in output
    assert "token_sha256:" in output
    assert "secret-refresh-A" not in output
    assert "secret-refresh-duplicate" not in output
    assert "secret-refresh-B" not in output


def test_plan_keeps_undecodable_and_unique_identities(tmp_path):
    root = tmp_path / ".hermes"
    account_a = _row("account-A", "account-A", "device_code", "account-A", "refresh-A")
    account_b = _row("account-B", "account-B", "manual:device_code", "account-B", "refresh-B")
    unknown_one = _row("unknown-1", "unknown-1", "manual:device_code", None, "refresh-X")
    unknown_two = _row("unknown-2", "unknown-2", "manual:device_code", None, "refresh-X")
    _write(root / "auth.json", _payload(account_a, [account_a, account_b, unknown_one, unknown_two]))

    plan = dedupe.build_plan(root)

    assert plan.files[0].removals == []


def test_plan_prefers_root_device_row_but_profile_freshest_row(tmp_path):
    root = tmp_path / ".hermes"
    device = _row("device", "primary", "device_code", "account-A", "refresh-old", exp=10)
    fresher_manual = _row(
        "manual",
        "duplicate",
        "manual:device_code",
        "account-A",
        "refresh-new",
        exp=999,
    )
    _write(root / "auth.json", _payload(device, [device, fresher_manual]))
    profile_path = root / "profiles" / "worker" / "auth.json"
    _write(profile_path, _payload(device, [fresher_manual, device]))

    plan = dedupe.build_plan(root)
    by_path = {item.path: item for item in plan.files}

    assert [item.entry_id for item in by_path[root / "auth.json"].removals] == ["manual"]
    assert [item.entry_id for item in by_path[profile_path].removals] == ["device"]


def test_plan_keeps_freshest_when_identity_is_not_root_singleton(tmp_path):
    root = tmp_path / ".hermes"
    singleton = _row("device", "primary", "device_code", "account-A", "refresh-A")
    older = _row("older-B", "older", "manual:device_code", "account-B", "refresh-B1", exp=10)
    newer = _row("newer-B", "newer", "manual:device_code", "account-B", "refresh-B2", exp=20)
    _write(root / "auth.json", _payload(singleton, [singleton, older, newer]))

    plan = dedupe.build_plan(root)

    assert [item.entry_id for item in plan.files[0].removals] == ["older-B"]
    assert plan.files[0].removals[0].survivor_id == "newer-B"


def test_apply_requires_explicit_yes(tmp_path):
    root = tmp_path / ".hermes"
    singleton = _row("device", "primary", "device_code", "account-A", "refresh-A")
    _write(root / "auth.json", _payload(singleton, [singleton]))

    with pytest.raises(SystemExit) as exc_info:
        dedupe.main(["--hermes-home", str(root), "--apply"])

    assert exc_info.value.code == 2


def test_apply_backs_up_each_changed_file_and_preserves_unrelated_data(tmp_path):
    root = tmp_path / ".hermes"
    device = _row(
        "device", "primary", "device_code", "account-A", "refresh-A", exp=20
    )
    duplicate = _row(
        "duplicate",
        "old label",
        "manual:device_code",
        "account-A",
        "refresh-A2",
        exp=10,
    )
    root_path = root / "auth.json"
    profile_path = root / "profiles" / "worker" / "auth.json"
    _write(root_path, _payload(device, [device, duplicate]))
    _write(profile_path, _payload(device, [duplicate, device]))

    assert dedupe.main(["--hermes-home", str(root), "--apply", "--yes"]) == 0

    for path in (root_path, profile_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert [row["id"] for row in payload["credential_pool"]["openai-codex"]] == ["device"]
        assert payload["providers"]["anthropic"]["api_key"] == "preserve-unrelated-provider"
        assert payload["credential_pool"]["anthropic"] == [{"id": "preserve-unrelated-row"}]
        backups = list(path.parent.glob("auth.json.bak-codex-pool-dedupe-*") )
        assert len(backups) == 1
        backup_payload = json.loads(backups[0].read_text(encoding="utf-8"))
        assert len(backup_payload["credential_pool"]["openai-codex"]) == 2


def test_apply_acquires_profile_lock_before_root_lock(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    device = _row("device", "primary", "device_code", "account-A", "refresh-A")
    duplicate = _row("duplicate", "duplicate", "manual:device_code", "account-A", "refresh-B")
    root_path = root / "auth.json"
    profile_path = root / "profiles" / "worker" / "auth.json"
    _write(root_path, _payload(device, [device, duplicate]))
    _write(profile_path, _payload(device, [device, duplicate]))
    acquired: list[Path] = []

    @contextlib.contextmanager
    def _record_lock(*, target_path):
        acquired.append(target_path)
        yield

    monkeypatch.setattr(dedupe, "_auth_store_lock", _record_lock)

    dedupe.apply_plan(dedupe.build_plan(root))

    assert acquired == [profile_path, root_path]


def test_apply_rejects_a_stale_plan_before_writing_backups(tmp_path):
    root = tmp_path / ".hermes"
    device = _row("device", "primary", "device_code", "account-A", "refresh-A")
    duplicate = _row("duplicate", "duplicate", "manual:device_code", "account-A", "refresh-B")
    root_path = root / "auth.json"
    _write(root_path, _payload(device, [device, duplicate]))
    plan = dedupe.build_plan(root)
    changed_payload = _payload(device, [device, duplicate])
    changed_payload["unrelated"] = "changed after planning"
    _write(root_path, changed_payload)

    with pytest.raises(RuntimeError, match="changed after planning"):
        dedupe.apply_plan(plan)

    assert not list(root.glob("auth.json.bak-codex-pool-dedupe-*"))
    assert json.loads(root_path.read_text(encoding="utf-8"))["unrelated"] == "changed after planning"


def test_apply_rolls_back_earlier_files_when_a_later_write_fails(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    device = _row("device", "primary", "device_code", "account-A", "refresh-A")
    duplicate = _row("duplicate", "duplicate", "manual:device_code", "account-A", "refresh-B")
    root_path = root / "auth.json"
    profile_path = root / "profiles" / "worker" / "auth.json"
    _write(root_path, _payload(device, [device, duplicate]))
    _write(profile_path, _payload(device, [device, duplicate]))
    originals = {
        root_path: root_path.read_bytes(),
        profile_path: profile_path.read_bytes(),
    }
    real_save = dedupe._save_auth_store
    calls = 0

    def fail_second_write(payload, target_path=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-write failure")
        return real_save(payload, target_path=target_path)

    monkeypatch.setattr(dedupe, "_save_auth_store", fail_second_write)

    with pytest.raises(RuntimeError, match="restored 2 attempted file"):
        dedupe.apply_plan(dedupe.build_plan(root))

    assert root_path.read_bytes() == originals[root_path]
    assert profile_path.read_bytes() == originals[profile_path]
    assert len(list(root_path.parent.glob("auth.json.bak-codex-pool-dedupe-*"))) == 1
    assert len(list(profile_path.parent.glob("auth.json.bak-codex-pool-dedupe-*"))) == 1
