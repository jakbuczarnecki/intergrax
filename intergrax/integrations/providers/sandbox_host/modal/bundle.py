# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_modal_sandbox_host as _legacy_create_modal_sandbox_host

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.sandbox_host.modal.integration import (
    MODAL_SANDBOX_HOST_PROVIDER_ID,
    ModalSandboxHostIntegration,
    ModalSandboxHostIntegrationConfig,
    ModalSandboxHostClient,
)

__all__ = [
    "create_modal_sandbox_host",
    "create_modal_sandbox_host_integration",
]


def create_modal_sandbox_host_integration(
    *,
    client: ModalSandboxHostClient | None = None,
    enabled: bool = False,
) -> ModalSandboxHostIntegration:
    """
    Build a contract-based Modal sandbox host integration.

    The legacy facade (create_modal_sandbox_host) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Modal sandbox host integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ModalSandboxHostIntegration.from_client(client, enabled=enabled)
    return ModalSandboxHostIntegration.for_provider(
        provider_id=MODAL_SANDBOX_HOST_PROVIDER_ID,
        display_name="Modal",
        config=ModalSandboxHostIntegrationConfig(enabled=enabled),
    )


def create_modal_sandbox_host(**kwargs: object) -> ModalSandboxHostIntegration:
    """Compatibility shim — constructs ModalSandboxHostIntegration from legacy runtime."""
    runtime = _legacy_create_modal_sandbox_host(**kwargs)
    if isinstance(runtime, ModalSandboxHostIntegration):
        return runtime
    return ModalSandboxHostIntegration.from_client(runtime)
