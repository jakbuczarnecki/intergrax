# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Modal sandbox host integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MODAL_SANDBOX_HOST_PROVIDER_ID = "modal"


class ModalSandboxHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Modal sandbox host integration."""

    pass


@runtime_checkable
class ModalSandboxHostClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ModalSandboxHostIntegration(SandboxHostIntegrationContract):
    """
    Modal sandbox host integration.

    The legacy facade (create_modal_sandbox_host) remains separate and backward-compatible.
    """

    config: ModalSandboxHostIntegrationConfig = ModalSandboxHostIntegrationConfig()
    _client: ModalSandboxHostClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ModalSandboxHostClient,
        *,
        enabled: bool = False,
    ) -> ModalSandboxHostIntegration:
        integration = cls.for_provider(
            provider_id=MODAL_SANDBOX_HOST_PROVIDER_ID,
            display_name="Modal",
            config=ModalSandboxHostIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ModalSandboxHostClient | None:
        return self._client
