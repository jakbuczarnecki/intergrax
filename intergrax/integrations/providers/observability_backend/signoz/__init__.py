# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SIGNOZ_OBSERVABILITY_PROVIDER_ID",
    "SignozObservabilityIntegration",
    "SignozObservabilityIntegrationConfig",
    "SignozObservabilityTransport",
    "create_signoz_observability_backend",
    "create_signoz_observability_integration",
    "register_signoz_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_signoz_observability_backend",
        "create_signoz_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SIGNOZ_OBSERVABILITY_PROVIDER_ID",
        "SignozObservabilityIntegration",
        "SignozObservabilityIntegrationConfig",
        "SignozObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_signoz_integration":
        from intergrax.integrations.providers.observability_backend.signoz.register import (
            register_signoz_integration,
        )

        return register_signoz_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.signoz import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.signoz import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
