# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Elasticsearch openers — internal to the elasticsearch integration package.

Only this module may construct ``httpx.Client`` / ``ElasticsearchRestClient`` for Elasticsearch.
All composition roots use ``bundle.create_elasticsearch_*`` or
``profile.resolve(OBSERVABILITY_BACKEND)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.elasticsearch.adapter import ElasticsearchObservabilityBackend
from intergrax.integrations.providers.observability_backend.elasticsearch.client import ElasticsearchRestClient
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    DEFAULT_TIMEOUT_SECONDS,
    ElasticsearchIntegrationConfig,
)


def _create_http_client(config: ElasticsearchIntegrationConfig) -> Any:
    import httpx

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"ApiKey {config.api_key}"
    auth = None
    if config.user and config.password:
        auth = (config.user, config.password)
    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return httpx.Client(
        base_url=config.api_base_url,
        timeout=timeout,
        headers=headers,
        auth=auth,
    )


def open_elasticsearch_rest_client(
    config: ElasticsearchIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ElasticsearchIntegrationConfig], Any]] = None,
) -> ElasticsearchRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return ElasticsearchRestClient(config, http_client=http_client)


def open_elasticsearch_observability_backend(
    config: ElasticsearchIntegrationConfig,
    *,
    implementation: Optional[ObservabilityBackend] = None,
    client: Optional[ElasticsearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ElasticsearchIntegrationConfig], Any]] = None,
) -> ObservabilityBackend:
    if implementation is not None:
        return implementation
    rest_client = client or open_elasticsearch_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return ElasticsearchObservabilityBackend(rest_client)
