# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "TRITON_VISION_SERVING_PROVIDER_ID",
    "TritonVisionServingIntegration",
    "TritonVisionServingIntegrationConfig",
    "TritonVisionServingClient",
    "create_triton_vision_serving",
    "create_triton_vision_serving_integration",
    "register_triton_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_triton_vision_serving",
        "create_triton_vision_serving_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "TRITON_VISION_SERVING_PROVIDER_ID",
        "TritonVisionServingIntegration",
        "TritonVisionServingIntegrationConfig",
        "TritonVisionServingClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "TRITON_VISION_SERVING_PROVIDER_ID",
        "TritonVisionServingIntegration",
        "TritonVisionServingIntegrationConfig",
        "TritonVisionServingClient",
    }
)

def __getattr__(name: str):
    if name == "register_triton_integration":
        from intergrax.integrations.providers.vision_serving.triton.register import register_triton_integration

        return register_triton_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vision_serving.triton import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vision_serving.triton import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vision_serving.triton import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
