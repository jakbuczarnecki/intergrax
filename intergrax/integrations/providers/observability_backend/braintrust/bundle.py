# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Braintrust integration bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.braintrust.adapter import BraintrustObservabilityBackend
from intergrax.integrations.providers.observability_backend.braintrust.client import BraintrustRestClient
from intergrax.integrations.providers.observability_backend.braintrust.config import BraintrustIntegrationConfig
from intergrax.integrations.providers.observability_backend.braintrust.opens import (
    open_braintrust_observability_backend,
    open_braintrust_rest_client,
)


@dataclass(frozen=True)
class BraintrustIntegrationBundle:
    config: BraintrustIntegrationConfig
    observability_backend: BraintrustObservabilityBackend
    rest_client: BraintrustRestClient


def create_braintrust_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[BraintrustRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[BraintrustIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> BraintrustIntegrationBundle:
    config = BraintrustIntegrationConfig.from_env(**config_overrides)
    rest_client = client or open_braintrust_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    backend = open_braintrust_observability_backend(
        config,
        implementation=observability_backend,
        client=rest_client,
    )
    assert isinstance(backend, BraintrustObservabilityBackend)
    return BraintrustIntegrationBundle(
        config=config,
        observability_backend=backend,
        rest_client=rest_client,
    )


def create_braintrust_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[BraintrustRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[BraintrustIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> BraintrustObservabilityBackend:
    """Catalog factory for ``IntegrationSlug.BRAINTRUST``."""
    return create_braintrust_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend
