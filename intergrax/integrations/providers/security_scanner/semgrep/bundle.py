# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_semgrep_security_scanner as _legacy_create_semgrep_security_scanner

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.security_scanner.semgrep.integration import (
    SEMGREP_SECURITY_SCANNER_PROVIDER_ID,
    SemgrepSecurityScannerIntegration,
    SemgrepSecurityScannerIntegrationConfig,
    SemgrepSecurityScannerClient,
)

__all__ = [
    "create_semgrep_security_scanner",
    "create_semgrep_security_scanner_integration",
]


def create_semgrep_security_scanner_integration(
    *,
    client: SemgrepSecurityScannerClient | None = None,
    enabled: bool = False,
) -> SemgrepSecurityScannerIntegration:
    """
    Build a contract-based Semgrep security scanner integration.

    The legacy facade (create_semgrep_security_scanner) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Semgrep security scanner integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SemgrepSecurityScannerIntegration.from_client(client, enabled=enabled)
    return SemgrepSecurityScannerIntegration.for_provider(
        provider_id=SEMGREP_SECURITY_SCANNER_PROVIDER_ID,
        display_name="Semgrep",
        config=SemgrepSecurityScannerIntegrationConfig(enabled=enabled),
    )


def create_semgrep_security_scanner(**kwargs: object) -> SemgrepSecurityScannerIntegration:
    """Compatibility shim — constructs SemgrepSecurityScannerIntegration from legacy runtime."""
    runtime = _legacy_create_semgrep_security_scanner(**kwargs)
    if isinstance(runtime, SemgrepSecurityScannerIntegration):
        return runtime
    return SemgrepSecurityScannerIntegration.from_client(runtime)
