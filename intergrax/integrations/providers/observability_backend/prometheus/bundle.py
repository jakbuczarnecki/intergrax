# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Prometheus integration bundle — the single composition root for Prometheus in Intergrax.

HTTP clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_prometheus_observability_backend()``, ``create_prometheus_integration()``, or
``profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.providers.observability_backend.prometheus.integration import (
    PrometheusObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.prometheus.client import PrometheusRestClient
from intergrax.integrations.providers.observability_backend.prometheus.config import PrometheusIntegrationConfig
from intergrax.integrations.providers.observability_backend.prometheus.opens import (
    open_prometheus_observability_backend,
    open_prometheus_rest_client,
)


@dataclass(frozen=True)
class PrometheusIntegrationBundle:
    config: PrometheusIntegrationConfig
    observability_backend: PrometheusObservabilityIntegration
    rest_client: PrometheusRestClient


def resolve_prometheus_config(**overrides: object) -> PrometheusIntegrationConfig:
    return PrometheusIntegrationConfig.from_env(**overrides)


def create_prometheus_integration(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[PrometheusRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[PrometheusIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> PrometheusIntegrationBundle:
    config = resolve_prometheus_config(**config_overrides)
    rest_client = client or open_prometheus_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    backend = open_prometheus_observability_backend(
        config,
        implementation=observability_backend,
        client=rest_client,
    )
    assert isinstance(backend, PrometheusObservabilityIntegration)
    return PrometheusIntegrationBundle(
        config=config,
        observability_backend=backend,
        rest_client=rest_client,
    )


def create_prometheus_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[PrometheusRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[PrometheusIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> PrometheusObservabilityIntegration:
    """Catalog factory for ``"prometheus"`` / ``OBSERVABILITY_BACKEND``."""
    return create_prometheus_integration(
        observability_backend=observability_backend,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).observability_backend


from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.prometheus.integration import (
    PROMETHEUS_OBSERVABILITY_PROVIDER_ID,
    PROMETHEUS_SUPPORTED_SIGNALS,
    PrometheusObservabilityIntegration,
    PrometheusObservabilityIntegrationConfig,
    PrometheusObservabilityTransport,
)


def create_prometheus_observability_integration(
    *,
    transport: PrometheusObservabilityTransport | None = None,
    enabled: bool = False,
) -> PrometheusObservabilityIntegration:
    """
    Build a contract-based Prometheus observability vendor integration.

    The legacy query facade (create_prometheus_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Prometheus observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return PrometheusObservabilityIntegration.from_transport(transport, enabled=enabled)
    return PrometheusObservabilityIntegration.for_provider(
        provider_id=PROMETHEUS_OBSERVABILITY_PROVIDER_ID,
        supported_signals=PROMETHEUS_SUPPORTED_SIGNALS,
        display_name="Prometheus",
        config=PrometheusObservabilityIntegrationConfig(enabled=enabled),
    )
