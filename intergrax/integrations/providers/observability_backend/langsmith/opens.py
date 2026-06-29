# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangSmith openers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.langsmith.client import LangSmithRestClient
from intergrax.integrations.providers.observability_backend.langsmith.config import DEFAULT_TIMEOUT_SECONDS, LangSmithIntegrationConfig
from intergrax.integrations.providers.observability_backend.langsmith.integration import (
    LangsmithObservabilityIntegration,
)


def _create_http_client(config: LangSmithIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return httpx.Client(
        base_url=config.base_url.rstrip("/"),
        headers={"x-api-key": config.api_key, "Accept": "application/json"},
        timeout=timeout,
    )


def open_langsmith_rest_client(
    config: LangSmithIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[LangSmithIntegrationConfig], Any]] = None,
) -> LangSmithRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return LangSmithRestClient(config, http_client=http_client)


def open_langsmith_observability_backend(
    config: LangSmithIntegrationConfig,
    *,
    implementation: Optional[ObservabilityBackend] = None,
    client: Optional[LangSmithRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[LangSmithIntegrationConfig], Any]] = None,
) -> ObservabilityBackend:
    if implementation is not None:
        return implementation
    rest_client = client or open_langsmith_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return LangsmithObservabilityIntegration.from_client(rest_client)
