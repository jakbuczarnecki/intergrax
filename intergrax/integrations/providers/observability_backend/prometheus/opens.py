# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Prometheus openers — internal to the prometheus integration package.

Only this module may construct ``httpx.Client`` / ``PrometheusRestClient`` for Prometheus.
All composition roots use ``bundle.create_prometheus_*`` or ``profile.resolve(OBSERVABILITY_BACKEND)``.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend._http_contract import (
    ObservabilityHttpClient,
    ObservabilityHttpClientFactory,
    ObservabilityHttpxClientAdapter,
)
from intergrax.integrations.providers.observability_backend.prometheus.integration import (
    PrometheusObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.prometheus.client import PrometheusRestClient
from intergrax.integrations.providers.observability_backend.prometheus.config import (
    DEFAULT_TIMEOUT_SECONDS,
    PrometheusIntegrationConfig,
)


def _create_http_client(config: PrometheusIntegrationConfig) -> ObservabilityHttpClient:
    import httpx

    headers = {"Accept": "application/json"}
    if config.bearer_token:
        headers["Authorization"] = f"Bearer {config.bearer_token}"
    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return ObservabilityHttpxClientAdapter(
        httpx.Client(
        base_url=config.api_base_url,
        timeout=timeout,
        headers=headers,
        )
    )


def open_prometheus_rest_client(
    config: PrometheusIntegrationConfig,
    *,
    http_client: ObservabilityHttpClient | None = None,
    http_client_factory: ObservabilityHttpClientFactory[PrometheusIntegrationConfig] | None = None,
) -> PrometheusRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return PrometheusRestClient(config, http_client=http_client)


def open_prometheus_observability_backend(
    config: PrometheusIntegrationConfig,
    *,
    implementation: Optional[ObservabilityBackend] = None,
    client: Optional[PrometheusRestClient] = None,
    http_client: ObservabilityHttpClient | None = None,
    http_client_factory: ObservabilityHttpClientFactory[PrometheusIntegrationConfig] | None = None,
) -> ObservabilityBackend:
    if implementation is not None:
        return implementation
    rest_client = client or open_prometheus_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return PrometheusObservabilityIntegration.from_client(rest_client)
