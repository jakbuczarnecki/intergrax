# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory helpers for OpenAI managed retrieval adapter."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.managed_retrieval import ManagedRetrievalBackend
from intergrax.integrations.providers.managed_retrieval.openai.adapter import (
    create_openai_managed_retrieval_adapter,
)
from intergrax.integrations.providers.managed_retrieval.openai.config import (
    OpenAIManagedRetrievalConfig,
    openai_managed_retrieval_config_from_env,
)
from intergrax.integrations.providers.managed_retrieval.openai.integration import (
    OPENAI_MANAGED_RETRIEVAL_PROVIDER_ID,
    OpenAIManagedRetrievalIntegration,
    OpenAIManagedRetrievalIntegrationConfig,
)

__all__ = [
    "create_openai_managed_retrieval",
    "create_openai_managed_retrieval_integration",
    "try_create_openai_managed_retrieval_from_env",
]


def create_openai_managed_retrieval_integration(
    *,
    client: ManagedRetrievalBackend | None = None,
    enabled: bool = False,
) -> OpenAIManagedRetrievalIntegration:
    """Build a contract-based OpenAI managed retrieval integration."""
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "OpenAI managed retrieval integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OpenAIManagedRetrievalIntegration.from_client(client, enabled=enabled)
    return OpenAIManagedRetrievalIntegration.for_provider(
        provider_id=OPENAI_MANAGED_RETRIEVAL_PROVIDER_ID,
        display_name="OpenAI",
        config=OpenAIManagedRetrievalIntegrationConfig(enabled=enabled),
    )


def create_openai_managed_retrieval(
    *,
    api_key: str | None = None,
    poll_interval_seconds: float | None = None,
    max_poll_attempts: int | None = None,
    **config_overrides: object,
) -> ManagedRetrievalBackend:
    """Catalog factory for ``openai`` / ``MANAGED_RETRIEVAL``."""
    overrides: dict[str, object] = dict(config_overrides)
    if api_key is not None:
        overrides["api_key"] = api_key
    if poll_interval_seconds is not None:
        overrides["poll_interval_seconds"] = poll_interval_seconds
    if max_poll_attempts is not None:
        overrides["max_poll_attempts"] = max_poll_attempts

    config = _resolve_openai_config(**overrides)
    adapter = create_openai_managed_retrieval_adapter(config)
    return OpenAIManagedRetrievalIntegration.from_client(adapter, enabled=True)


def try_create_openai_managed_retrieval_from_env() -> ManagedRetrievalBackend | None:
    config = openai_managed_retrieval_config_from_env()
    if config is None:
        return None
    return create_openai_managed_retrieval(
        api_key=config.api_key,
        poll_interval_seconds=config.poll_interval_seconds,
        max_poll_attempts=config.max_poll_attempts,
    )


def _resolve_openai_config(**overrides: object) -> OpenAIManagedRetrievalConfig:
    config = openai_managed_retrieval_config_from_env()
    if config is None:
        api_key = str(overrides.get("api_key", "")).strip()
        if not api_key:
            raise IntegrationConfigurationError(
                "OpenAI managed retrieval requires OPENAI_API_KEY or explicit api_key",
            )
        poll_raw = overrides.get("poll_interval_seconds", 5.0)
        attempts_raw = overrides.get("max_poll_attempts", 120)
        return OpenAIManagedRetrievalConfig(
            api_key=api_key,
            poll_interval_seconds=float(poll_raw),
            max_poll_attempts=int(attempts_raw),
        )

    data = {
        "api_key": config.api_key,
        "poll_interval_seconds": config.poll_interval_seconds,
        "max_poll_attempts": config.max_poll_attempts,
    }
    data.update(overrides)
    return OpenAIManagedRetrievalConfig(
        api_key=str(data["api_key"]).strip(),
        poll_interval_seconds=float(data["poll_interval_seconds"]),
        max_poll_attempts=int(data["max_poll_attempts"]),
    )
