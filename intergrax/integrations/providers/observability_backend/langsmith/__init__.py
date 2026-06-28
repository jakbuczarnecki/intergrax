# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LANGSMITH_OBSERVABILITY_PROVIDER_ID",
    "LangsmithObservabilityIntegration",
    "LangsmithObservabilityIntegrationConfig",
    "LangsmithObservabilityTransport",
    "create_langsmith_observability_backend",
    "create_langsmith_observability_integration",
    "register_langsmith_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_langsmith_observability_backend",
        "create_langsmith_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LANGSMITH_OBSERVABILITY_PROVIDER_ID",
        "LangsmithObservabilityIntegration",
        "LangsmithObservabilityIntegrationConfig",
        "LangsmithObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_langsmith_integration":
        from intergrax.integrations.providers.observability_backend.langsmith.register import (
            register_langsmith_integration,
        )

        return register_langsmith_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.langsmith import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.langsmith import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
