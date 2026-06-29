# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Braintrust openers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.braintrust.adapter import _BraintrustObservabilityBackend
from intergrax.integrations.providers.observability_backend.braintrust.client import BraintrustRestClient
from intergrax.integrations.providers.observability_backend.braintrust.config import BraintrustIntegrationConfig


def _create_http_client(config: BraintrustIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or 30.0)
    return httpx.Client(
        base_url=config.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {config.api_key}", "Accept": "application/json"},
        timeout=timeout,
    )


def open_braintrust_rest_client(
    config: BraintrustIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[BraintrustIntegrationConfig], Any]] = None,
) -> BraintrustRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return BraintrustRestClient(config, http_client=http_client)


def open_braintrust_observability_backend(
    config: BraintrustIntegrationConfig,
    *,
    implementation: Optional[ObservabilityBackend] = None,
    client: Optional[BraintrustRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[BraintrustIntegrationConfig], Any]] = None,
) -> ObservabilityBackend:
    if implementation is not None:
        return implementation
    rest_client = client or open_braintrust_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return BraintrustObservabilityBackend(rest_client)
