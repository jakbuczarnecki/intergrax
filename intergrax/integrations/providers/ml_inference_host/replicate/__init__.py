# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID",
    "ReplicateMlInferenceHostIntegration",
    "ReplicateMlInferenceHostIntegrationConfig",
    "ReplicateMlInferenceHostClient",
    "create_replicate_ml_inference_host",
    "create_replicate_ml_inference_host_integration",
    "register_replicate_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_replicate_ml_inference_host",
        "create_replicate_ml_inference_host_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID",
        "ReplicateMlInferenceHostIntegration",
        "ReplicateMlInferenceHostIntegrationConfig",
        "ReplicateMlInferenceHostClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID",
        "ReplicateMlInferenceHostIntegration",
        "ReplicateMlInferenceHostIntegrationConfig",
        "ReplicateMlInferenceHostClient",
    }
)

def __getattr__(name: str):
    if name == "register_replicate_integration":
        from intergrax.integrations.providers.ml_inference_host.replicate.register import register_replicate_integration

        return register_replicate_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.ml_inference_host.replicate import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ml_inference_host.replicate import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ml_inference_host.replicate import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
