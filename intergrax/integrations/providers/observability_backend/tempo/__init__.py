# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "TEMPO_OBSERVABILITY_PROVIDER_ID",
    "TempoObservabilityIntegration",
    "TempoObservabilityIntegrationConfig",
    "TempoObservabilityTransport",
    "create_tempo_observability_backend",
    "create_tempo_observability_integration",
    "register_tempo_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_tempo_observability_backend",
        "create_tempo_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "TEMPO_OBSERVABILITY_PROVIDER_ID",
        "TempoObservabilityIntegration",
        "TempoObservabilityIntegrationConfig",
        "TempoObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_tempo_integration":
        from intergrax.integrations.providers.observability_backend.tempo.register import (
            register_tempo_integration,
        )

        return register_tempo_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.tempo import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.tempo import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
