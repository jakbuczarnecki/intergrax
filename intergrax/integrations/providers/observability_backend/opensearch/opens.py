# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenSearch openers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.opensearch.client import OpenSearchRestClient
from intergrax.integrations.providers.observability_backend.opensearch.config import OpenSearchIntegrationConfig
from intergrax.integrations.providers.observability_backend.opensearch.integration import (
    OpensearchObservabilityIntegration,
)


def _create_http_client(config: OpenSearchIntegrationConfig) -> Any:
    import httpx

    auth = None
    headers: dict[str, str] = {"Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"ApiKey {config.api_key}"
    elif config.user and config.password:
        auth = (config.user, config.password)
    timeout = float(config.timeout_seconds or 30.0)
    return httpx.Client(
        base_url=config.base_url.rstrip("/"),
        auth=auth,
        headers=headers,
        timeout=timeout,
    )


def open_opensearch_rest_client(
    config: OpenSearchIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[OpenSearchIntegrationConfig], Any]] = None,
) -> OpenSearchRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return OpenSearchRestClient(config, http_client=http_client)


def open_opensearch_observability_backend(
    config: OpenSearchIntegrationConfig,
    *,
    implementation: Optional[ObservabilityBackend] = None,
    client: Optional[OpenSearchRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[OpenSearchIntegrationConfig], Any]] = None,
) -> ObservabilityBackend:
    if implementation is not None:
        return implementation
    rest_client = client or open_opensearch_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return OpensearchObservabilityIntegration.from_client(rest_client)
