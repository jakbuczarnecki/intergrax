# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.langsmith.client import LangSmithRestClient
from intergrax.integrations.providers.observability_backend.langsmith.config import LangSmithIntegrationConfig
from intergrax.integrations.providers.observability_backend.langsmith.integration import (
    LANGSMITH_OBSERVABILITY_PROVIDER_ID,
    LANGSMITH_SUPPORTED_SIGNALS,
    LangsmithObservabilityIntegration,
    LangsmithObservabilityIntegrationConfig,
    LangsmithObservabilityTransport,
)
from intergrax.integrations.providers.observability_backend.langsmith.opens import (
    open_langsmith_observability_backend,
    open_langsmith_rest_client,
)

__all__ = [
    "LangsmithIntegrationBundle",
    "create_langsmith_observability_backend",
    "create_langsmith_observability_integration",
    "create_langsmith_integration",
    "resolve_langsmith_config",
]


@dataclass(frozen=True)
class LangsmithIntegrationBundle:
    config: LangSmithIntegrationConfig
    observability_backend: LangsmithObservabilityIntegration
    rest_client: LangSmithRestClient


def resolve_langsmith_config(**overrides: object) -> LangSmithIntegrationConfig:
    return LangSmithIntegrationConfig.from_env(**overrides)


def create_langsmith_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[LangSmithRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[LangSmithIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> LangsmithIntegrationBundle:
    config = resolve_langsmith_config(**config_overrides)
    rest_client = client or open_langsmith_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    backend = open_langsmith_observability_backend(
        config,
        implementation=observability_backend,
        client=rest_client,
    )
    assert isinstance(backend, LangsmithObservabilityIntegration)
    return LangsmithIntegrationBundle(
        config=config,
        observability_backend=backend,
        rest_client=rest_client,
    )


def create_langsmith_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[LangSmithRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[LangSmithIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> LangsmithObservabilityIntegration:
    """Catalog factory for ``"langsmith"`` / ``OBSERVABILITY_BACKEND``."""
    return create_langsmith_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend


def create_langsmith_observability_integration(
    *,
    transport: LangsmithObservabilityTransport | None = None,
    enabled: bool = False,
) -> LangsmithObservabilityIntegration:
    """
    Build a contract-based Langsmith observability vendor integration.

    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Langsmith observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return LangsmithObservabilityIntegration.from_transport(transport, enabled=enabled)
    return LangsmithObservabilityIntegration.for_provider(
        provider_id=LANGSMITH_OBSERVABILITY_PROVIDER_ID,
        supported_signals=LANGSMITH_SUPPORTED_SIGNALS,
        display_name="Langsmith",
        config=LangsmithObservabilityIntegrationConfig(enabled=enabled),
    )
