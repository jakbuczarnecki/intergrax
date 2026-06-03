# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenSearch integration bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.opensearch.adapter import OpenSearchObservabilityBackend
from intergrax.integrations.providers.observability_backend.opensearch.client import OpenSearchRestClient
from intergrax.integrations.providers.observability_backend.opensearch.config import OpenSearchIntegrationConfig
from intergrax.integrations.providers.observability_backend.opensearch.opens import (
    open_opensearch_observability_backend,
    open_opensearch_rest_client,
)


@dataclass(frozen=True)
class OpenSearchIntegrationBundle:
    config: OpenSearchIntegrationConfig
    observability_backend: OpenSearchObservabilityBackend
    rest_client: OpenSearchRestClient


def create_opensearch_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[OpenSearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[OpenSearchIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> OpenSearchIntegrationBundle:
    config = OpenSearchIntegrationConfig.from_env(**config_overrides)
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
    assert isinstance(backend, OpenSearchObservabilityBackend)
    return OpenSearchIntegrationBundle(
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
) -> OpenSearchObservabilityBackend:
    """Catalog factory for ``"opensearch"``."""
    return create_opensearch_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend
