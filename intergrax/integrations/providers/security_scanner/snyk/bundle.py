# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_snyk_security_scanner

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.security_scanner.snyk.integration import (
    SNYK_SECURITY_SCANNER_PROVIDER_ID,
    SnykSecurityScannerIntegration,
    SnykSecurityScannerIntegrationConfig,
    SnykSecurityScannerClient,
)

__all__ = [
    "create_snyk_security_scanner",
    "create_snyk_security_scanner_integration",
]


def create_snyk_security_scanner_integration(
    *,
    client: SnykSecurityScannerClient | None = None,
    enabled: bool = False,
) -> SnykSecurityScannerIntegration:
    """
    Build a contract-based Snyk security scanner integration.

    The legacy facade (create_snyk_security_scanner) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Snyk security scanner integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SnykSecurityScannerIntegration.from_client(client, enabled=enabled)
    return SnykSecurityScannerIntegration.for_provider(
        provider_id=SNYK_SECURITY_SCANNER_PROVIDER_ID,
        display_name="Snyk",
        config=SnykSecurityScannerIntegrationConfig(enabled=enabled),
    )
