# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SEMGREP_SECURITY_SCANNER_PROVIDER_ID",
    "SemgrepSecurityScannerIntegration",
    "SemgrepSecurityScannerIntegrationConfig",
    "SemgrepSecurityScannerClient",
    "create_semgrep_security_scanner",
    "create_semgrep_security_scanner_integration",
    "register_semgrep_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_semgrep_security_scanner",
        "create_semgrep_security_scanner_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SEMGREP_SECURITY_SCANNER_PROVIDER_ID",
        "SemgrepSecurityScannerIntegration",
        "SemgrepSecurityScannerIntegrationConfig",
        "SemgrepSecurityScannerClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SEMGREP_SECURITY_SCANNER_PROVIDER_ID",
        "SemgrepSecurityScannerIntegration",
        "SemgrepSecurityScannerIntegrationConfig",
        "SemgrepSecurityScannerClient",
    }
)

def __getattr__(name: str):
    if name == "register_semgrep_integration":
        from intergrax.integrations.providers.security_scanner.semgrep.register import register_semgrep_integration

        return register_semgrep_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.security_scanner.semgrep import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.security_scanner.semgrep import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.security_scanner.semgrep import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
