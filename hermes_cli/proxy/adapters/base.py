"""Abstract base for proxy upstream adapters; the proxy server is otherwise provider-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, FrozenSet, Optional


@dataclass(frozen=True)
class UpstreamCredential:
    """A resolved bearer + base URL ready to forward to."""

    bearer: str  # token only, no ``Bearer`` prefix
    base_url: str  # e.g. ``https://inference-api.nousresearch.com/v1``
    token_type: str = "Bearer"
    expires_at: Optional[str] = None  # ISO-8601, informational


class UpstreamAdapter(ABC):
    """Contract for an upstream provider the proxy can forward to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter key used on the CLI (e.g. ``"nous"``)."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name for logs and ``proxy status``."""

    @property
    @abstractmethod
    def allowed_paths(self) -> FrozenSet[str]:
        """Paths relative to the proxy's ``/v1`` mount (``"/chat/completions"`` ⇒
        ``/v1/chat/completions``); anything else gets a 404 with a helpful body."""

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Cheap (no network) usable-credentials check; ``proxy start`` uses it for a clear
        up-front error before binding a port."""

    @abstractmethod
    def get_credential(self, *, model: Optional[str] = None) -> UpstreamCredential:
        """Fresh credential (refreshing/rotating + persisting as needed). Raises RuntimeError when
        unauthenticated or refresh fails; the proxy then returns 401 to the client."""

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
        model: Optional[str] = None,
    ) -> Optional[UpstreamCredential]:
        """Alternate credential for a one-shot retry after the upstream rejects the first request;
        default is no retry."""
        _ = failed_credential, status_code, model
        return None

    def raw_chat_route_for_model(self, model: Optional[str]) -> Optional[dict[str, str]]:
        """Return a raw-provider compatibility route for a model, if needed."""
        _ = model
        return None

    def image_generation_route_for_model(self, model: Optional[str]) -> Optional[dict[str, str]]:
        """Return an image-provider route for OpenAI Images compatibility."""
        _ = model
        return None

    def list_models(self) -> Optional[list[dict[str, Any]]]:
        """Return an aggregated model list, or ``None`` to forward upstream."""
        return None

    def describe(self) -> str:
        """One-line status summary for ``proxy status``."""
        try:
            cred = self.get_credential()
        except Exception as exc:  # pragma: no cover - defensive
            return f"{self.display_name}: not ready ({exc})"
        ttl = f" (expires {cred.expires_at})" if cred.expires_at else ""
        return f"{self.display_name}: {cred.base_url}{ttl}"


__all__ = ["UpstreamAdapter", "UpstreamCredential"]
