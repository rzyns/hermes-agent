"""Regression tests for Nous Portal inference_base_url host-allowlist validation.

A poisoned ``inference_base_url`` from a Portal refresh response (network
MITM, malicious response injection) would otherwise be persisted to
auth.json and forwarded with the user's legitimate invoke JWT
bearer on every subsequent proxy request, exfiltrating their inference
budget and opening a response-injection channel into the IDE / chat
client. ``_validate_nous_inference_url_from_network()`` blocks any URL
outside the allowlist at the source.

These tests verify:

1. The validator's host + scheme rules.
2. Refresh and runtime resolution reject untrusted Portal URLs.
3. The proxy adapter applies the validator as belt-and-suspenders.
4. The env-var override path (``NOUS_INFERENCE_BASE_URL``) is NOT
   gated by the validator — that's the documented dev/staging escape
   hatch.
"""

from __future__ import annotations

import logging

from hermes_cli.auth import (
    DEFAULT_NOUS_INFERENCE_URL,
    _ALLOWED_NOUS_INFERENCE_HOSTS,
    _validate_nous_inference_url_from_network,
)


class TestValidatorRules:


    def test_attacker_host_rejected(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            assert (
                _validate_nous_inference_url_from_network("https://attacker.com/v1")
                is None
            )
        assert any("attacker.com" in rec.message for rec in caplog.records)



    def test_default_inference_url_is_in_allowlist(self):
        """Sanity check: DEFAULT_NOUS_INFERENCE_URL must itself validate.

        If anyone retargets the default away from
        ``inference-api.nousresearch.com``, they MUST update the allowlist
        in the same change — otherwise the allowlist would reject the
        Portal's own legitimate default and break every install.
        """
        assert (
            _validate_nous_inference_url_from_network(DEFAULT_NOUS_INFERENCE_URL)
            == DEFAULT_NOUS_INFERENCE_URL.rstrip("/")
        )



class TestHealsPoisonedStoredValue:
    """A stored inference_base_url that is NOT in the allowlist (e.g. a
    stale ``stg-inference-api.nousresearch.com`` persisted before the
    allowlist existed) must be HEALED back to the production default on
    the next refresh — not silently retained.

    Before the fix, the refresh sites only assigned the validated URL
    ``if refreshed_url:`` and otherwise left the poisoned value in place,
    so the "falling back to default" warning was logged but never
    actually took effect — every subsequent call kept hitting the dead
    staging endpoint (real incident: opus-4.8 routed to nous, nous pinned
    to staging, every request + the aux compression call 401'd).
    """

    def test_refresh_resets_rejected_url_to_default(self, monkeypatch):
        import hermes_cli.auth as auth

        poisoned = "https://stg-inference-api.nousresearch.com/v1"
        state = {
            "access_token": "tok",
            "refresh_token": "rtok",
            "client_id": "hermes-cli",
            "portal_base_url": auth.DEFAULT_NOUS_PORTAL_URL,
            "inference_base_url": poisoned,
        }

        # Force the refresh branch and return another rejected (staging) URL,
        # exercising the validator-returns-None heal path.
        monkeypatch.setattr(auth, "_nous_invoke_jwt_status", lambda *a, **k: "needs_refresh")
        monkeypatch.setattr(
            auth,
            "_refresh_access_token",
            lambda **k: {
                "access_token": "newtok",
                "refresh_token": "newrtok",
                "expires_in": 3600,
                "inference_base_url": poisoned,  # Portal still hands back staging
            },
        )
        # Skip the JWT usability assertions (orthogonal to URL healing).
        monkeypatch.setattr(auth, "_assert_nous_inference_jwt_usable", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_select_nous_invoke_jwt", lambda *a, **k: None)

        result = auth.refresh_nous_oauth_from_state(state, force_refresh=True)

        assert result["inference_base_url"] == auth.DEFAULT_NOUS_INFERENCE_URL, (
            "rejected Portal URL must heal to the production default, "
            f"got {result['inference_base_url']!r}"
        )


class TestEnvOverrideWins:
    """``NOUS_INFERENCE_BASE_URL`` must win over the stored value for the
    URL used to build the inference client / returned to callers.

    This is the documented dev/staging escape hatch. The breakage it
    regresses against: the security allowlist (#30611) plus the refresh
    heal (#49735) mean a staging login's stored ``inference_base_url`` is
    rejected and rewritten to the production default, and the runtime
    resolver previously read that stored (prod) value *before* the env
    var — so an OAuth user could not reach staging at all, even with the
    env override set. The override is consulted FIRST here, while the
    PERSISTED value stays the validated, network-provenance one (the env
    override is a runtime overlay, never written to auth.json).
    """

    STAGING = "https://stg-inference-api.nousresearch.com/v1"

    def _patch_no_refresh(self, monkeypatch, auth, state):
        import contextlib

        # No refresh fires: the stored access token is a usable invoke JWT.
        monkeypatch.setattr(auth, "_nous_invoke_jwt_status", lambda *a, **k: None)
        monkeypatch.setattr(
            auth, "_auth_store_lock", lambda *a, **k: contextlib.nullcontext()
        )
        monkeypatch.setattr(auth, "_load_auth_store", lambda *a, **k: {})
        monkeypatch.setattr(auth, "_load_provider_state", lambda store, pid: state)
        monkeypatch.setattr(
            auth,
            "_load_provider_state_with_source",
            lambda store, pid: (state, None),
        )
        monkeypatch.setattr(auth, "_save_provider_state", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_save_provider_state_to_source", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_save_auth_store", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_write_shared_nous_state", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_sync_nous_pool_from_auth_store", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_resolve_verify", lambda *a, **k: True)
        monkeypatch.setattr(auth, "_assert_nous_inference_jwt_usable", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_select_nous_invoke_jwt", lambda *a, **k: None)

    def _base_state(self, auth, stored):
        return {
            "access_token": "tok",
            "refresh_token": "rtok",
            "client_id": "hermes-cli",
            "portal_base_url": auth.DEFAULT_NOUS_PORTAL_URL,
            "inference_base_url": stored,
            "agent_key": "ak-123",
        }


    def test_no_refresh_env_override_not_persisted(self, monkeypatch):
        """The env override is a runtime overlay: it must never be written
        back into the stored state (auth.json)."""
        import hermes_cli.auth as auth

        state = self._base_state(auth, auth.DEFAULT_NOUS_INFERENCE_URL)
        self._patch_no_refresh(monkeypatch, auth, state)
        monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", self.STAGING)

        auth.resolve_nous_runtime_credentials()

        assert state["inference_base_url"] == auth.DEFAULT_NOUS_INFERENCE_URL, (
            "env override leaked into persisted state — it must stay a "
            f"runtime overlay, got {state['inference_base_url']!r}"
        )


    def test_no_refresh_heals_poisoned_stored_without_env(self, monkeypatch):
        """A poisoned stored staging host (persisted before the allowlist)
        still heals to the default when no env override is present — the
        #50265 no-refresh-read-path heal, folded in here."""
        import hermes_cli.auth as auth

        state = self._base_state(auth, self.STAGING)
        self._patch_no_refresh(monkeypatch, auth, state)
        monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)

        result = auth.resolve_nous_runtime_credentials()
        assert result["base_url"] == auth.DEFAULT_NOUS_INFERENCE_URL, (
            "poisoned stored URL must heal to the production default on the "
            f"no-refresh read path, got {result['base_url']!r}"
        )



class TestProxyAdapterEnvOverride:
    """The Nous proxy adapter is the second chokepoint: it re-validates the
    base_url returned by resolve_nous_runtime_credentials() against the prod
    allowlist. That re-validation must not clobber a legitimate
    NOUS_INFERENCE_BASE_URL staging override.
    """

    STAGING = "https://stg-inference-api.nousresearch.com/v1"

    def _adapter_with_runtime(self, monkeypatch, *, base_url: str):
        import hermes_cli.proxy.adapters.nous_portal as nous_adapter

        adapter = nous_adapter.NousPortalAdapter()
        monkeypatch.setattr(adapter, "_read_state", lambda: {"access_token": "stored"})
        monkeypatch.setattr(
            nous_adapter,
            "resolve_nous_runtime_credentials",
            lambda **kwargs: {"api_key": "invoke-token", "base_url": base_url},
        )
        return adapter

    def test_proxy_adapter_consults_env_override(self, monkeypatch):
        adapter = self._adapter_with_runtime(
            monkeypatch,
            base_url="https://attacker.example/v1",
        )
        monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", self.STAGING)

        credential = adapter._get_credential()

        assert credential.base_url == self.STAGING

    def test_proxy_adapter_rejects_untrusted_resolved_url(self, monkeypatch):
        import hermes_cli.auth as auth

        adapter = self._adapter_with_runtime(
            monkeypatch,
            base_url="https://attacker.example/v1",
        )
        monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)

        credential = adapter._get_credential()

        assert credential.base_url == auth.DEFAULT_NOUS_INFERENCE_URL
