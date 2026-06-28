# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SNYK_SECURITY_SCANNER_PROVIDER_ID",
    "SnykSecurityScannerIntegration",
    "SnykSecurityScannerIntegrationConfig",
    "SnykSecurityScannerClient",
    "create_snyk_security_scanner",
    "create_snyk_security_scanner_integration",
    "register_snyk_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_snyk_security_scanner",
        "create_snyk_security_scanner_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SNYK_SECURITY_SCANNER_PROVIDER_ID",
        "SnykSecurityScannerIntegration",
        "SnykSecurityScannerIntegrationConfig",
        "SnykSecurityScannerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SNYK_SECURITY_SCANNER_PROVIDER_ID",
        "SnykSecurityScannerIntegration",
        "SnykSecurityScannerIntegrationConfig",
        "SnykSecurityScannerClient",
    }
)

def __getattr__(name: str):
    if name == "register_snyk_integration":
        from intergrax.integrations.providers.security_scanner.snyk.register import register_snyk_integration

        return register_snyk_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.security_scanner.snyk import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.security_scanner.snyk import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.security_scanner.snyk import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
