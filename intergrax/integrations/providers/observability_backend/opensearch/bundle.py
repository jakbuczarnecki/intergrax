# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.opensearch.client import OpenSearchRestClient
from intergrax.integrations.providers.observability_backend.opensearch.config import OpenSearchIntegrationConfig
from intergrax.integrations.providers.observability_backend.opensearch.integration import (
    OPENSEARCH_OBSERVABILITY_PROVIDER_ID,
    OPENSEARCH_SUPPORTED_SIGNALS,
    OpensearchObservabilityIntegration,
    OpensearchObservabilityIntegrationConfig,
    OpensearchObservabilityTransport,
)
from intergrax.integrations.providers.observability_backend.opensearch.opens import (
    open_opensearch_observability_backend,
    open_opensearch_rest_client,
)

__all__ = [
    "OpensearchIntegrationBundle",
    "create_opensearch_observability_backend",
    "create_opensearch_observability_integration",
    "create_opensearch_integration",
    "resolve_opensearch_config",
]


@dataclass(frozen=True)
class OpensearchIntegrationBundle:
    config: OpenSearchIntegrationConfig
    observability_backend: OpensearchObservabilityIntegration
    rest_client: OpenSearchRestClient


def resolve_opensearch_config(**overrides: object) -> OpenSearchIntegrationConfig:
    return OpenSearchIntegrationConfig.from_env(**overrides)


def create_opensearch_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[OpenSearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[OpenSearchIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> OpensearchIntegrationBundle:
    config = resolve_opensearch_config(**config_overrides)
    rest_client = client or open_opensearch_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    backend = open_opensearch_observability_backend(
        config,
        implementation=observability_backend,
        client=rest_client,
    )
    assert isinstance(backend, OpensearchObservabilityIntegration)
    return OpensearchIntegrationBundle(
        config=config,
        observability_backend=backend,
        rest_client=rest_client,
    )


def create_opensearch_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[OpenSearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[OpenSearchIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> OpensearchObservabilityIntegration:
    """Catalog factory for ``"opensearch"`` / ``OBSERVABILITY_BACKEND``."""
    return create_opensearch_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend


def create_opensearch_observability_integration(
    *,
    transport: OpensearchObservabilityTransport | None = None,
    enabled: bool = False,
) -> OpensearchObservabilityIntegration:
    """
    Build a contract-based OpenSearch observability vendor integration.

    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "OpenSearch observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return OpensearchObservabilityIntegration.from_transport(transport, enabled=enabled)
    return OpensearchObservabilityIntegration.for_provider(
        provider_id=OPENSEARCH_OBSERVABILITY_PROVIDER_ID,
        supported_signals=OPENSEARCH_SUPPORTED_SIGNALS,
        display_name="OpenSearch",
        config=OpensearchObservabilityIntegrationConfig(enabled=enabled),
    )
