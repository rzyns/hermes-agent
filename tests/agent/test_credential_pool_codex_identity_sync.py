"""Regression tests for identity-safe OpenAI Codex pool synchronization."""

from __future__ import annotations

import base64
import json

from hermes_cli.auth import AuthError


_ACCOUNT_CLAIM = "https://api.openai.com/auth"


def _jwt(account_id: str) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return (
        f"{_part({'alg': 'none', 'typ': 'JWT'})}."
        f"{_part({_ACCOUNT_CLAIM: {'chatgpt_account_id': account_id}})}.sig"
    )


def _row(
    entry_id: str,
    source: str,
    account_id: str | None,
    refresh_token: str,
) -> dict:
    access_token = _jwt(account_id) if account_id is not None else "not-a-jwt"
    return {
        "id": entry_id,
        "label": entry_id,
        "source": source,
        "auth_type": "oauth",
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


def _store(singleton: dict, rows: list[dict]) -> dict:
    return {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": singleton["access_token"],
                    "refresh_token": singleton["refresh_token"],
                },
                "last_refresh": "2026-09-03T00:00:00Z",
            }
        },
        "credential_pool": {"openai-codex": rows},
    }


def _write_store(hermes_home, payload: dict) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")


def _load_pool(tmp_path, monkeypatch, payload: dict):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _write_store(hermes_home, payload)

    from agent.credential_pool import load_pool

    return hermes_home, load_pool("openai-codex")


def _entry(pool, entry_id: str):
    return next(item for item in pool.entries() if item.id == entry_id)


def test_codex_sync_does_not_clobber_independent_manual_entry(tmp_path, monkeypatch):
    account_a_old = _row("singleton", "device_code", "account-A", "refresh-A-old")
    account_a_new = _row("singleton", "device_code", "account-A", "refresh-A-new")
    account_b = _row("manual-B", "manual:device_code", "account-B", "refresh-B")
    hermes_home, pool = _load_pool(
        tmp_path,
        monkeypatch,
        _store(account_a_old, [account_a_old, account_b]),
    )

    _write_store(hermes_home, _store(account_a_new, [account_a_old, account_b]))

    manual_before = _entry(pool, "manual-B")
    manual_after = pool._sync_codex_entry_from_auth_store(manual_before)
    device_after = pool._sync_codex_entry_from_auth_store(_entry(pool, "singleton"))

    assert manual_after is manual_before
    assert manual_after.access_token == account_b["access_token"]
    assert manual_after.refresh_token == "refresh-B"
    assert device_after.access_token == account_a_new["access_token"]
    assert device_after.refresh_token == "refresh-A-new"


def test_codex_sync_adopts_same_account_manual_entry(tmp_path, monkeypatch):
    account_a_old = _row("manual-A", "manual:device_code", "account-A", "refresh-A-old")
    account_a_new = _row("singleton", "device_code", "account-A", "refresh-A-new")
    _hermes_home, pool = _load_pool(
        tmp_path,
        monkeypatch,
        _store(account_a_new, [account_a_old]),
    )

    updated = pool._sync_codex_entry_from_auth_store(_entry(pool, "manual-A"))

    assert updated.access_token == account_a_new["access_token"]
    assert updated.refresh_token == "refresh-A-new"


def test_codex_sync_does_not_adopt_undecodable_manual_entry(tmp_path, monkeypatch):
    singleton = _row("singleton", "device_code", "account-A", "refresh-A-new")
    undecodable = _row("manual-unknown", "manual:device_code", None, "refresh-unknown")
    _hermes_home, pool = _load_pool(
        tmp_path,
        monkeypatch,
        _store(singleton, [singleton, undecodable]),
    )

    before = _entry(pool, "manual-unknown")
    after = pool._sync_codex_entry_from_auth_store(before)

    assert after is before
    assert after.access_token == "not-a-jwt"
    assert after.refresh_token == "refresh-unknown"


def test_codex_proactive_pre_refresh_sync_preserves_independent_manual_entry(
    tmp_path, monkeypatch
):
    singleton = _row("singleton", "device_code", "account-A", "refresh-A-new")
    independent = _row("manual-B", "manual:device_code", "account-B", "refresh-B")
    _hermes_home, pool = _load_pool(
        tmp_path,
        monkeypatch,
        _store(singleton, [singleton, independent]),
    )
    refresh_calls: list[tuple[str, str]] = []

    def _refresh(access_token: str, refresh_token: str) -> dict:
        refresh_calls.append((access_token, refresh_token))
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "last_refresh": "2026-09-03T00:01:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", _refresh)

    updated = pool._refresh_entry(_entry(pool, "manual-B"), force=True)

    assert refresh_calls == [(independent["access_token"], "refresh-B")]
    assert updated is not None
    assert updated.access_token == independent["access_token"]
    assert updated.refresh_token == "refresh-B"


def test_codex_post_401_resync_preserves_independent_manual_entry(
    tmp_path, monkeypatch
):
    account_b = _row("singleton", "device_code", "account-B", "refresh-B")
    manual_b = _row("manual-B", "manual:device_code", "account-B", "refresh-B")
    account_a = _row("singleton", "device_code", "account-A", "refresh-A-new")
    hermes_home, pool = _load_pool(
        tmp_path,
        monkeypatch,
        _store(account_b, [account_b, manual_b]),
    )

    def _refresh_then_401(_access_token: str, _refresh_token: str) -> dict:
        payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
        payload["providers"]["openai-codex"]["tokens"] = {
            "access_token": account_a["access_token"],
            "refresh_token": account_a["refresh_token"],
        }
        _write_store(hermes_home, payload)
        raise AuthError(
            "token invalidated",
            provider="openai-codex",
            code="codex_refresh_failed",
            relogin_required=False,
        )

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", _refresh_then_401)

    pool._refresh_entry(_entry(pool, "manual-B"), force=True)
    manual_after = _entry(pool, "manual-B")

    assert manual_after.access_token == manual_b["access_token"]
    assert manual_after.refresh_token == "refresh-B"
