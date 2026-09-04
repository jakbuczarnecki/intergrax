# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Semgrep security scanner."""

from __future__ import annotations

from intergrax.integrations.providers.security_scanner.semgrep.bundle import (
    create_semgrep_security_scanner_integration,
)
from intergrax.integrations.providers.security_scanner.semgrep.integration import (
    SEMGREP_SECURITY_SCANNER_PROVIDER_ID,
    SemgrepSecurityScannerIntegration,
    SemgrepSecurityScannerIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="security_scanner",
    provider_id=SEMGREP_SECURITY_SCANNER_PROVIDER_ID,
    integration_class=SemgrepSecurityScannerIntegration,
    contract_class=SecurityScannerIntegrationContract,
    contract_factory=create_semgrep_security_scanner_integration,
    display_name="Semgrep",
    config_class=SemgrepSecurityScannerIntegrationConfig,
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
