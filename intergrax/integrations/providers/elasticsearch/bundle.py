# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Elasticsearch integration bundle — the single composition root for Elasticsearch in Intergrax.

HTTP clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_elasticsearch_observability_backend()``, ``create_elasticsearch_integration()``, or
``profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.elasticsearch.adapter import ElasticsearchObservabilityBackend
from intergrax.integrations.providers.elasticsearch.client import ElasticsearchRestClient
from intergrax.integrations.providers.elasticsearch.config import ElasticsearchIntegrationConfig
from intergrax.integrations.providers.elasticsearch.opens import (
    open_elasticsearch_observability_backend,
    open_elasticsearch_rest_client,
)


@dataclass(frozen=True)
class ElasticsearchIntegrationBundle:
    config: ElasticsearchIntegrationConfig
    observability_backend: ElasticsearchObservabilityBackend
    rest_client: ElasticsearchRestClient


def resolve_elasticsearch_config(**overrides: object) -> ElasticsearchIntegrationConfig:
    return ElasticsearchIntegrationConfig.from_env(**overrides)


def create_elasticsearch_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[ElasticsearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ElasticsearchIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> ElasticsearchIntegrationBundle:
    config = resolve_elasticsearch_config(**config_overrides)
    rest_client = client or open_elasticsearch_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    backend = open_elasticsearch_observability_backend(
        config,
        implementation=observability_backend,
        client=rest_client,
    )
    assert isinstance(backend, ElasticsearchObservabilityBackend)
    return ElasticsearchIntegrationBundle(
        config=config,
        observability_backend=backend,
        rest_client=rest_client,
    )


def create_elasticsearch_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[ElasticsearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ElasticsearchIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> ElasticsearchObservabilityBackend:
    """Catalog factory for ``IntegrationSlug.ELASTICSEARCH`` / ``OBSERVABILITY_BACKEND``."""
    return create_elasticsearch_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend
