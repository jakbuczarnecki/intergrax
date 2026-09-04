# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Trivy security scanner."""

from __future__ import annotations

from intergrax.integrations.providers.security_scanner.trivy.bundle import (
    create_trivy_security_scanner_integration,
)
from intergrax.integrations.providers.security_scanner.trivy.integration import (
    TRIVY_SECURITY_SCANNER_PROVIDER_ID,
    TrivySecurityScannerIntegration,
    TrivySecurityScannerIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="security_scanner",
    provider_id=TRIVY_SECURITY_SCANNER_PROVIDER_ID,
    integration_class=TrivySecurityScannerIntegration,
    contract_class=SecurityScannerIntegrationContract,
    contract_factory=create_trivy_security_scanner_integration,
    display_name="Trivy",
    config_class=TrivySecurityScannerIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
