"""OpenAI Codex (Responses API) provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

openai_codex = ProviderProfile(
    name="openai-codex",
    aliases=("codex", "openai_codex"),
    api_mode="codex_responses",
    env_vars=(),  # OAuth external — no API key
    base_url="https://chatgpt.com/backend-api/codex",
    auth_type="oauth_external",
    # gpt-5.5 is the current canonical model for this endpoint. The codebase
    # already treats it as the Codex family via _is_codex_gpt55() (compaction
    # threshold, context cap at 272K). Earlier the default was left empty to
    # avoid silent drift (gpt-5.3-codex -> gpt-5.2-codex -> gpt-5.4), but the
    # family is now stable enough to pin. If OpenAI shifts the accepted slug
    # again, update this (and _is_codex_gpt55).
    default_aux_model="gpt-5.5",
)

register_provider(openai_codex)
