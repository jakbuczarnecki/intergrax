# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Snyk security scanner."""

from __future__ import annotations

from intergrax.integrations.providers.security_scanner.snyk.bundle import (
    create_snyk_security_scanner_integration,
)
from intergrax.integrations.providers.security_scanner.snyk.integration import (
    SNYK_SECURITY_SCANNER_PROVIDER_ID,
    SnykSecurityScannerIntegration,
    SnykSecurityScannerIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="security_scanner",
    provider_id=SNYK_SECURITY_SCANNER_PROVIDER_ID,
    integration_class=SnykSecurityScannerIntegration,
    contract_class=SecurityScannerIntegrationContract,
    contract_factory=create_snyk_security_scanner_integration,
    display_name="Snyk",
    config_class=SnykSecurityScannerIntegrationConfig,
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
