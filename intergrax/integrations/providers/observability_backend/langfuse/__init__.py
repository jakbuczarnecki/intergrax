# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LANGFUSE_OBSERVABILITY_PROVIDER_ID",
    "LangfuseObservabilityIntegration",
    "LangfuseObservabilityIntegrationConfig",
    "LangfuseObservabilityTransport",
    "create_langfuse_observability_backend",
    "create_langfuse_observability_integration",
    "register_langfuse_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_langfuse_observability_backend",
        "create_langfuse_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LANGFUSE_OBSERVABILITY_PROVIDER_ID",
        "LangfuseObservabilityIntegration",
        "LangfuseObservabilityIntegrationConfig",
        "LangfuseObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_langfuse_integration":
        from intergrax.integrations.providers.observability_backend.langfuse.register import (
            register_langfuse_integration,
        )

        return register_langfuse_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.langfuse import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.langfuse import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
