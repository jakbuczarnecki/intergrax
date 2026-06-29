# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_trivy_security_scanner as _legacy_create_trivy_security_scanner

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.security_scanner.trivy.integration import (
    TRIVY_SECURITY_SCANNER_PROVIDER_ID,
    TrivySecurityScannerIntegration,
    TrivySecurityScannerIntegrationConfig,
    TrivySecurityScannerClient,
)

__all__ = [
    "create_trivy_security_scanner",
    "create_trivy_security_scanner_integration",
]


def create_trivy_security_scanner_integration(
    *,
    client: TrivySecurityScannerClient | None = None,
    enabled: bool = False,
) -> TrivySecurityScannerIntegration:
    """
    Build a contract-based Trivy security scanner integration.

    The legacy facade (create_trivy_security_scanner) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Trivy security scanner integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TrivySecurityScannerIntegration.from_client(client, enabled=enabled)
    return TrivySecurityScannerIntegration.for_provider(
        provider_id=TRIVY_SECURITY_SCANNER_PROVIDER_ID,
        display_name="Trivy",
        config=TrivySecurityScannerIntegrationConfig(enabled=enabled),
    )


def create_trivy_security_scanner(**kwargs: object) -> TrivySecurityScannerIntegration:
    """Compatibility shim — constructs TrivySecurityScannerIntegration from legacy runtime."""
    runtime = _legacy_create_trivy_security_scanner(**kwargs)
    if isinstance(runtime, TrivySecurityScannerIntegration):
        return runtime
    return TrivySecurityScannerIntegration.from_client(runtime)
