# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MODAL_SANDBOX_HOST_PROVIDER_ID",
    "ModalSandboxHostIntegration",
    "ModalSandboxHostIntegrationConfig",
    "ModalSandboxHostClient",
    "create_modal_sandbox_host",
    "create_modal_sandbox_host_integration",
    "register_modal_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_modal_sandbox_host",
        "create_modal_sandbox_host_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MODAL_SANDBOX_HOST_PROVIDER_ID",
        "ModalSandboxHostIntegration",
        "ModalSandboxHostIntegrationConfig",
        "ModalSandboxHostClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MODAL_SANDBOX_HOST_PROVIDER_ID",
        "ModalSandboxHostIntegration",
        "ModalSandboxHostIntegrationConfig",
        "ModalSandboxHostClient",
    }
)

def __getattr__(name: str):
    if name == "register_modal_integration":
        from intergrax.integrations.providers.sandbox_host.modal.register import register_modal_integration

        return register_modal_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.sandbox_host.modal import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.sandbox_host.modal import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.sandbox_host.modal import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
