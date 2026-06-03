# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangSmith integration bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.langsmith.adapter import LangSmithObservabilityBackend
from intergrax.integrations.providers.observability_backend.langsmith.client import LangSmithRestClient
from intergrax.integrations.providers.observability_backend.langsmith.config import LangSmithIntegrationConfig
from intergrax.integrations.providers.observability_backend.langsmith.opens import (
    open_langsmith_observability_backend,
    open_langsmith_rest_client,
)


@dataclass(frozen=True)
class LangSmithIntegrationBundle:
    config: LangSmithIntegrationConfig
    observability_backend: LangSmithObservabilityBackend
    rest_client: LangSmithRestClient


def create_langsmith_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[LangSmithRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[LangSmithIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> LangSmithIntegrationBundle:
    config = LangSmithIntegrationConfig.from_env(**config_overrides)
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
    assert isinstance(backend, LangSmithObservabilityBackend)
    return LangSmithIntegrationBundle(
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
) -> LangSmithObservabilityBackend:
    """Catalog factory for ``"langsmith"``."""
    return create_langsmith_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend
