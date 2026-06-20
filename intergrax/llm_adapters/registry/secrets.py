# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Resolve LLM API keys for Tier-3 hosts (env + optional secrets map).

Tier-3 applications SHOULD load vault/Integration ``secrets_store`` values into
``secrets`` before ``LLMProfile.create_adapter()`` — never commit raw keys.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Mapping, Optional

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

if TYPE_CHECKING:
    from intergrax.integrations.contracts.secrets_store import SecretsStore

# Provider slug -> primary API key environment variable
_API_KEY_ENV: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.CLAUDE: "ANTHROPIC_API_KEY",
    LLMProvider.GEMINI: "GOOGLE_API_KEY",
    LLMProvider.MISTRAL: "MISTRAL_API_KEY",
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.TOGETHER: "TOGETHER_API_KEY",
    LLMProvider.FIREWORKS: "FIREWORKS_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    LLMProvider.XAI: "XAI_API_KEY",
    LLMProvider.COHERE: "COHERE_API_KEY",
    LLMProvider.COHERE_NATIVE: "COHERE_API_KEY",
    LLMProvider.AZURE_AI_INFERENCE: "AZURE_AI_INFERENCE_API_KEY",
}


def _provider_slug(provider: LLMProvider | str) -> str:
    if isinstance(provider, LLMProvider):
        return provider.value
    return str(provider or "").strip().lower()


def _coerce_builtin_provider(provider: LLMProvider | str) -> LLMProvider | None:
    if isinstance(provider, LLMProvider):
        return provider
    try:
        return LLMProvider(_provider_slug(provider))
    except ValueError:
        return None


def api_key_env_for_provider(provider: LLMProvider | str) -> Optional[str]:
    builtin = _coerce_builtin_provider(provider)
    if builtin is None:
        return None
    return _API_KEY_ENV.get(builtin)


def resolve_api_key(
    provider: LLMProvider | str,
    secrets: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """
    Resolve API key: explicit ``secrets['api_key']`` or provider env var.
    """
    if secrets and secrets.get("api_key"):
        return str(secrets["api_key"]).strip()
    builtin = _coerce_builtin_provider(provider)
    env_name = api_key_env_for_provider(builtin) if builtin is not None else None
    if env_name:
        val = os.getenv(env_name)
        if val and val.strip():
            return val.strip()
    return None


def default_secret_path_for_provider(provider: LLMProvider | str, *, prefix: str = "llm") -> str:
    """Vault-style path: ``{prefix}/{provider}/api_key``."""
    return f"{prefix.strip('/')}/{_provider_slug(provider)}/api_key"


def load_api_key_from_secrets_store(
    store: SecretsStore,
    provider: LLMProvider | str,
    *,
    path: Optional[str] = None,
) -> str:
    """Read provider API key from Integration ``SecretsStore`` (rotation-friendly)."""
    secret_path = path or default_secret_path_for_provider(provider)
    value = store.get_secret(secret_path)
    if not value or not str(value).strip():
        raise RuntimeError(
            f"Empty secret at path='{secret_path}' for provider='{_provider_slug(provider)}'."
        )
    return str(value).strip()


def merge_secrets_into_options(
    provider: LLMProvider | str,
    options: dict,
    secrets: Optional[Mapping[str, str]] = None,
) -> dict:
    """Copy resolved ``api_key`` into adapter constructor kwargs when applicable."""
    merged = dict(options)
    key = resolve_api_key(provider, secrets)
    if key and "api_key" not in merged:
        merged["api_key"] = key
    return merged
