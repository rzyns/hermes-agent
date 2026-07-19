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
from typing import Any, FrozenSet, Optional

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)


_ALLOWED_PATHS: FrozenSet[str] = frozenset({
    "/chat/completions",
    "/responses",
    "/completions",
    "/embeddings",
    "/images/generations",
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


def _any_provider_has_credentials() -> bool:
    """Return True if any provider has stored credentials that can be used."""
    try:
        from hermes_cli.auth import _load_auth_store, _auth_store_lock
        with _auth_store_lock():
            store = _load_auth_store()
        providers = store.get("providers") or {}
        for provider_id, state in providers.items():
            if not isinstance(state, dict):
                continue
            # Check for the most common credential shapes.
            if (
                state.get("access_token")
                or state.get("agent_key")
                or state.get("tokens")
            ):
                return True
        return False
    except Exception as exc:
        logger.debug(
            "subscription-proxy: could not check credential availability: %s", exc
        )
        return False


def _provider_has_credentials(provider_id: str) -> bool:
    """Return True if a specific provider has stored credentials."""
    try:
        from hermes_cli.auth import _load_auth_store, _auth_store_lock
        with _auth_store_lock():
            store = _load_auth_store()
        providers = store.get("providers") or {}
        state = providers.get(provider_id)
        if not isinstance(state, dict):
            return False
        return bool(
            state.get("access_token")
            or state.get("agent_key")
            or state.get("tokens")
        )
    except Exception as exc:
        logger.debug(
            "subscription-proxy: could not check credentials for %r: %s",
            provider_id,
            exc,
        )
        return False


def _bare_model_name(model: Optional[str]) -> str:
    """Return a normalized model slug without provider prefix."""
    if not isinstance(model, str):
        return ""
    return model.strip().lower().rsplit("/", 1)[-1]


def _provider_candidates_for_model(model: Optional[str]) -> list[str]:
    """Preferred provider order for a requested model family."""
    bare = _bare_model_name(model)
    if not bare:
        return []
    if bare == "gpt-5.5" or bare.startswith("gpt-"):
        # OpenAI Codex OAuth only serves GPT-family models; do not let a
        # MiniMax credential win simply because it appears first in auth.json.
        return ["openai-codex", "nous"]
    if bare.startswith("minimax"):
        return ["minimax-oauth"]
    if bare.startswith("glm-"):
        return ["zai"]
    # Fall back to searching credentialed providers' model lists.
    # This keeps /v1/models and routing self-consistent: any model that
    # list_models() advertises is guaranteed to be routable.
    return _providers_owning_model(model)


# Static model lists for providers whose profile doesn't list them.
# Single source of truth used by both list_models() and routing lookup.
_STATIC_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai-codex": ["gpt-5.5"],
    "minimax-oauth": [
        "MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
        "MiniMax-M2.5", "MiniMax-M2.5-highspeed",
        "MiniMax-M2.1", "MiniMax-M2.1-highspeed", "MiniMax-M2",
    ],
}


def _providers_owning_model(model: Optional[str]) -> list[str]:
    """Return credentialed providers whose model list contains ``model``.

    Searches the same sources as ``list_models()``: ``fallback_models``,
    ``default_aux_model``, and the static model table.  Only returns providers
    that actually have credentials (auth.json or env var).
    """
    if not isinstance(model, str) or not model.strip():
        return []

    import os

    target = model.strip()
    target_lower = target.lower()
    found: list[str] = []

    try:
        import providers as _providers_mod

        for profile in _providers_mod.list_providers():
            name = profile.name
            has_auth = _provider_has_credentials(name)
            env_has = any(os.getenv(v) for v in profile.env_vars) if profile.env_vars else False
            if not (has_auth or env_has):
                continue

            # Build this provider's known model set
            known: set[str] = set()
            known.update(m.lower() for m in (profile.fallback_models or []))
            aux = getattr(profile, "default_aux_model", "")
            if aux:
                known.add(aux.lower())
            known.update(m.lower() for m in _STATIC_PROVIDER_MODELS.get(name, []))

            if target_lower in known:
                found.append(name)
    except Exception as exc:
        logger.debug("subscription-proxy: model ownership lookup failed: %s", exc)

    return found


def _raw_chat_route_for_model(model: Optional[str]) -> Optional[dict[str, str]]:
    """Return a raw-provider chat route for models that need protocol translation."""
    bare = _bare_model_name(model)
    if bare == "gpt-5.5" or bare.startswith("gpt-"):
        # The chatgpt.com Codex backend is Responses-only.  Hermes' shared raw
        # provider handler wraps it with a Chat Completions-compatible surface.
        if _provider_has_credentials("openai-codex"):
            return {"provider": "openai-codex", "model": bare}
    return None


def _image_generation_route_for_model(
    model: Optional[str],
) -> Optional[dict[str, str]]:
    """Map OpenAI image model IDs to the Codex image-provider quality tiers."""
    bare = _bare_model_name(model)
    if bare == "gpt-image-2":
        bare = "gpt-image-2-medium"
    if bare in {
        "gpt-image-2-low",
        "gpt-image-2-medium",
        "gpt-image-2-high",
    }:
        return {"provider": "openai-codex", "model": bare}
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
        if active:
            return True
        # Auto-detect: if any provider has usable credentials, we're good.
        return _any_provider_has_credentials()

    def get_credential(self, *, model: Optional[str] = None) -> UpstreamCredential:
        preferred = _provider_candidates_for_model(model)
        if preferred:
            return self._credential_from_preferred_providers(preferred, model=model)

        # Try Nous Portal first — it has dedicated JWT refresh logic.
        if _is_nous_auth_available():
            with self._lock:
                return self._credential_from_nous()
        # Try the explicitly active provider next.
        active = _get_active_provider_id()
        if active:
            with self._lock:
                return self._credential_from_active_provider(active, requested_model=model)
        # Fall back to auto-detection.
        with self._lock:
            return self._credential_from_auto_detected_provider(model=model)

    def raw_chat_route_for_model(self, model: Optional[str]) -> Optional[dict[str, str]]:
        return _raw_chat_route_for_model(model)

    def image_generation_route_for_model(
        self,
        model: Optional[str],
    ) -> Optional[dict[str, str]]:
        return _image_generation_route_for_model(model)

    def list_models(self) -> list[dict[str, Any]]:
        """Aggregate models from all credentialed providers.

        Returns a flat list of OpenAI-shaped model objects, merging:
        - provider ``fallback_models`` lists
        - provider ``default_aux_model`` values
        - static known-good models for providers with empty profiles
        """
        import os
        import time

        import providers as _providers_mod

        seen: set[str] = set()
        models: list[dict[str, Any]] = []
        created = int(time.time())

        for profile in _providers_mod.list_providers():
            name = profile.name
            has_auth = _provider_has_credentials(name)
            env_has = any(os.getenv(v) for v in profile.env_vars) if profile.env_vars else False
            if not (has_auth or env_has):
                continue

            # Collect model IDs from fallback_models, default_aux_model, and static list.
            candidates: list[str] = []
            candidates.extend(profile.fallback_models or [])
            aux = getattr(profile, "default_aux_model", "")
            if aux:
                candidates.append(aux)
            candidates.extend(_STATIC_PROVIDER_MODELS.get(name, []))

            for model_id in candidates:
                if model_id in seen:
                    continue
                seen.add(model_id)
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": name,
                })

        return models

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
        model: Optional[str] = None,
    ) -> Optional[UpstreamCredential]:
        _ = model
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

    def _credential_from_preferred_providers(
        self,
        providers: list[str],
        *,
        model: Optional[str],
    ) -> UpstreamCredential:
        """Resolve credentials from model-specific provider candidates only."""
        last_error: Optional[Exception] = None
        for provider_id in providers:
            try:
                if provider_id == "nous":
                    if not _is_nous_auth_available():
                        continue
                    with self._lock:
                        return self._credential_from_nous()
                with self._lock:
                    return self._credential_from_active_provider(
                        provider_id,
                        requested_model=model,
                    )
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "subscription-proxy: model %r skipped provider %r: %s",
                    model,
                    provider_id,
                    exc,
                )
                continue
        if last_error is not None:
            raise RuntimeError(
                f"No usable provider found for requested model {model!r}. "
                f"Last error: {last_error}"
            ) from last_error
        raise RuntimeError(
            f"No configured provider is known to serve requested model {model!r}."
        )

    def _credential_from_active_provider(
        self,
        provider: Optional[str] = None,
        *,
        requested_model: Optional[str] = None,
    ) -> UpstreamCredential:
        """Resolve credentials for an explicitly named (or active) provider.

        Uses ``resolve_provider_client`` to get a configured client, then
        extracts the bearer token and base URL for the proxy forwarder.
        """
        active_provider = provider or _get_active_provider_id()
        if not active_provider:
            raise RuntimeError(
                "No active subscription provider found. "
                "Run `hermes auth add <provider>` to configure one."
            )

        try:
            from agent.auxiliary_client import _to_openai_base_url, resolve_provider_client

            # Thread-safe resolution — returns (client, resolved_model).
            client, _resolved_model = resolve_provider_client(
                active_provider,
                model=requested_model,
            )
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
        # Note: AnthropicAuxiliaryClient exposes "https://api.minimax.io/anthropic"
        # as its base_url, but the actual OpenAI-compatible upstream path is
        # "https://api.minimax.io/v1" (it rewrites /anthropic → /v1 internally via
        # _to_openai_base_url).  Apply the same rewrite here so the proxy produces
        # the correct upstream URL (e.g. "https://api.minimax.io/v1" +
        # "/chat/completions" = "https://api.minimax.io/v1/chat/completions").
        if base_url is not None:
            base_url = _to_openai_base_url(str(base_url))
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

    def _credential_from_auto_detected_provider(
        self,
        *,
        model: Optional[str] = None,
    ) -> UpstreamCredential:
        """Auto-detect the best available provider and return its credentials.

        Tries providers in order of preference (minimax-oauth, openai-codex,
        etc.) by attempting to resolve a client for each and returning the first
        that succeeds. This handles the case where no provider is explicitly
        set as active but valid credentials exist in the auth store.
        """
        # Order matters.  Model-specific candidates are handled first so a
        # gpt-* request does not get sent to MiniMax just because MiniMax has
        # valid credentials.
        candidates = _provider_candidates_for_model(model) or [
            "minimax-oauth",
            "openai-codex",
        ]

        last_error: Optional[RuntimeError] = None
        for provider_id in candidates:
            try:
                cred = self._credential_from_active_provider(
                    provider_id,
                    requested_model=model,
                )
                logger.info(
                    "subscription-proxy: auto-detected provider %r for proxy",
                    provider_id,
                )
                return cred
            except Exception as exc:
                logger.debug(
                    "subscription-proxy: auto-detect skipped %r: %s",
                    provider_id,
                    exc,
                )
                if isinstance(exc, RuntimeError):
                    last_error = exc
                continue

        if last_error:
            raise last_error
        raise RuntimeError(
            "No usable subscription credentials found. "
            "Run `hermes auth add <provider>` to configure a provider."
        )


__all__ = ["SubscriptionProxyAdapter"]
