# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Grafana observability backend."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.grafana.bundle import (
    create_grafana_observability_integration,
)
from intergrax.integrations.providers.observability_backend.grafana.integration import (
    GRAFANA_OBSERVABILITY_PROVIDER_ID,
    GrafanaObservabilityIntegration,
    GrafanaObservabilityIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories import OBSERVABILITY_VENDOR_INTEGRATION_KIND
from intergrax.runtime.integrations.observability import ObservabilityVendorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="observability_backend",
    provider_id=GRAFANA_OBSERVABILITY_PROVIDER_ID,
    integration_class=GrafanaObservabilityIntegration,
    contract_class=ObservabilityVendorIntegrationContract,
    contract_factory=create_grafana_observability_integration,
    integration_kind=OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    display_name="Grafana",
    config_class=GrafanaObservabilityIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.EXPORT,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
