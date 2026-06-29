# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.braintrust.client import BraintrustRestClient
from intergrax.integrations.providers.observability_backend.braintrust.config import BraintrustIntegrationConfig
from intergrax.integrations.providers.observability_backend.braintrust.integration import (
    BRAINTRUST_OBSERVABILITY_PROVIDER_ID,
    BRAINTRUST_SUPPORTED_SIGNALS,
    BraintrustObservabilityIntegration,
    BraintrustObservabilityIntegrationConfig,
    BraintrustObservabilityTransport,
)
from intergrax.integrations.providers.observability_backend.braintrust.opens import (
    open_braintrust_observability_backend,
    open_braintrust_rest_client,
)

__all__ = [
    "BraintrustIntegrationBundle",
    "create_braintrust_observability_backend",
    "create_braintrust_observability_integration",
    "create_braintrust_integration",
    "resolve_braintrust_config",
]


@dataclass(frozen=True)
class BraintrustIntegrationBundle:
    config: BraintrustIntegrationConfig
    observability_backend: BraintrustObservabilityIntegration
    rest_client: BraintrustRestClient


def resolve_braintrust_config(**overrides: object) -> BraintrustIntegrationConfig:
    return BraintrustIntegrationConfig.from_env(**overrides)


def create_braintrust_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[BraintrustRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[BraintrustIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> BraintrustIntegrationBundle:
    config = resolve_braintrust_config(**config_overrides)
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
    assert isinstance(backend, BraintrustObservabilityIntegration)
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
) -> BraintrustObservabilityIntegration:
    """Catalog factory for ``"braintrust"`` / ``OBSERVABILITY_BACKEND``."""
    return create_braintrust_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend


def create_braintrust_observability_integration(
    *,
    transport: BraintrustObservabilityTransport | None = None,
    enabled: bool = False,
) -> BraintrustObservabilityIntegration:
    """
    Build a contract-based Braintrust observability vendor integration.

    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Braintrust observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return BraintrustObservabilityIntegration.from_transport(transport, enabled=enabled)
    return BraintrustObservabilityIntegration.for_provider(
        provider_id=BRAINTRUST_OBSERVABILITY_PROVIDER_ID,
        supported_signals=BRAINTRUST_SUPPORTED_SIGNALS,
        display_name="Braintrust",
        config=BraintrustObservabilityIntegrationConfig(enabled=enabled),
    )
