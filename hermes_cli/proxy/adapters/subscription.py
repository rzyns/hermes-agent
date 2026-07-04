"""Subscription-proxy upstream adapter.

Proxies to any configured OAuth provider by detecting the active subscription
from the user's auth store and resolving a fresh bearer + base URL per request.

Unlike ``NousPortalAdapter`` (which hard-codes the Nous inference endpoint) or
``XAIGrokAdapter`` (which only handles xAI), this adapter routes to whichever
provider the user has an active subscription for — as identified by the auth
store and the Subscription Proxy Protocol.
"""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Optional

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_ALLOWED_PATHS: FrozenSet[str] = frozenset({
    "/chat/completions",
    "/responses",
    "/completions",
    "/embeddings",
    "/models",
})


def _is_nous_auth_available() -> bool:
    """Check whether Nous Portal auth state is present and non-empty."""
    try:
        from hermes_cli.auth import _load_auth_store, _auth_store_lock
        with _auth_store_lock():
            store = _load_auth_store()
        providers = store.get("providers") or {}
        nous = providers.get("nous")
        if not isinstance(nous, dict):
            return False
        return bool(
            nous.get("agent_key")
            or (nous.get("refresh_token") and nous.get("access_token"))
        )
    except Exception as exc:
        logger.debug("subscription-proxy: could not check Nous auth state: %s", exc)
        return False


def _get_active_provider_id() -> Optional[str]:
    """Return the provider id for the active subscription, if any."""
    try:
        from hermes_cli.auth import get_active_provider
        return get_active_provider()
    except Exception as exc:
        logger.debug("subscription-proxy: could not detect active provider: %s", exc)
        return None


class SubscriptionProxyAdapter(UpstreamAdapter):
    """Proxy upstream that routes to the user's active subscription provider.

    Detects the active provider from the auth store and uses
    ``resolve_provider_client`` to obtain a correctly-configured client,
    then extracts the bearer token and base URL for the proxy forwarder.

    If the active provider is Nous Portal, uses the dedicated
    ``resolve_nous_runtime_credentials`` path for correct JWT handling
    (including force-refresh on 401 from the upstream).
    """

    auth_hint = "hermes auth add nous  (or your OAuth provider's auth command)"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "subscription"

    @property
    def display_name(self) -> str:
        return "Subscription Proxy"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        # At least one auth path must be usable.
        if _is_nous_auth_available():
            return True
        # Also accept the active provider being set (even if non-Nous).
        active = _get_active_provider_id()
        return active is not None

    def get_credential(self) -> UpstreamCredential:
        # Try Nous Portal first — it has dedicated JWT refresh logic.
        if _is_nous_auth_available():
            with self._lock:
                return self._credential_from_nous()
        # Fall back to the generic provider-client path.
        with self._lock:
            return self._credential_from_active_provider()

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        # Only retry on 401 (auth expired); 429 must be handled by the
        # individual pool/rotation logic inside each provider's client.
        if status_code != 401:
            return None

        # If the failed credential came from Nous, force-refresh the JWT.
        if _is_nous_auth_available():
            try:
                with self._lock:
                    cred = self._credential_from_nous(force_refresh=True)
                if cred.bearer != failed_credential.bearer:
                    return cred
            except Exception as exc:
                logger.warning(
                    "subscription-proxy: retry Nous JWT refresh failed: %s", exc
                )
        return None

    def _credential_from_nous(
        self,
        *,
        force_refresh: bool = False,
    ) -> UpstreamCredential:
        from hermes_cli.auth import (
            DEFAULT_NOUS_INFERENCE_URL,
            resolve_nous_runtime_credentials,
            _quarantine_nous_pool_entries,
            _auth_store_lock,
            _load_auth_store,
            _save_auth_store,
            _write_shared_nous_state,
        )

        try:
            refreshed = resolve_nous_runtime_credentials(force_refresh=force_refresh)
        except Exception as exc:
            # Terminal refresh failure — quarantine so future proxy starts
            # don't keep hammering a known-bad token.
            try:
                with _auth_store_lock():
                    store = _load_auth_store()
                    providers = store.get("providers") or {}
                    state = providers.get("nous")
                    if isinstance(state, dict):
                        _quarantine_nous_pool_entries(
                            store, exc, reason="proxy_refresh_failure"
                        )
                        providers["nous"] = state
                        _save_auth_store(store)
                        _write_shared_nous_state(state)
            except Exception:
                pass  # Best-effort quarantine
            raise RuntimeError(
                f"Failed to refresh Nous Portal credentials: {exc}"
            ) from exc

        runtime_key = refreshed.get("api_key")
        if not runtime_key:
            raise RuntimeError(
                "Nous Portal refresh did not return a usable inference JWT. "
                "Try `hermes auth add nous` to re-authenticate."
            )

        base_url = (
            refreshed.get("base_url") or DEFAULT_NOUS_INFERENCE_URL
        ).rstrip("/")

        return UpstreamCredential(
            bearer=runtime_key,
            base_url=base_url,
            expires_at=refreshed.get("expires_at"),
        )

    def _credential_from_active_provider(self) -> UpstreamCredential:
        """Resolve credentials for the active (non-Nous) provider.

        Uses ``resolve_provider_client`` to get a configured client, then
        extracts the bearer token and base URL for the proxy forwarder.
        """
        active_provider = _get_active_provider_id()
        if not active_provider:
            raise RuntimeError(
                "No active subscription provider found. "
                "Run `hermes auth add <provider>` to configure one."
            )

        try:
            from agent.auxiliary_client import resolve_provider_client

            # Thread-safe resolution — returns (client, resolved_model).
            client, _resolved_model = resolve_provider_client(active_provider)
            if client is None:
                raise RuntimeError(
                    f"Provider {active_provider!r} is not configured or not available. "
                    "Run `hermes auth add <provider>` to configure it."
                )
        except ImportError as exc:
            raise RuntimeError(
                f"Could not resolve provider {active_provider!r}: {exc}"
            ) from exc

        # Extract base URL and bearer from the client.
        base_url = getattr(client, "base_url", None)
        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        if not base_url:
            raise RuntimeError(
                f"Provider {active_provider!r} client has no base_url. "
                "Check your model_routes configuration."
            )

        api_key = getattr(client, "api_key", None)
        if api_key is None:
            raise RuntimeError(
                f"Provider {active_provider!r} client has no api_key. "
                "Check your auth configuration."
            )
        bearer = str(api_key).strip()
        if not bearer:
            raise RuntimeError(
                f"Provider {active_provider!r} api_key is empty. "
                "Re-authenticate with `hermes auth add {active_provider}`."
            )

        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url,
            expires_at=None,
        )


__all__ = ["SubscriptionProxyAdapter"]
