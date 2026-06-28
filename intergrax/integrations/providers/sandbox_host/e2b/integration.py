# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""E2B sandbox host integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

E2B_SANDBOX_HOST_PROVIDER_ID = "e2b"


class E2bSandboxHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for E2B sandbox host integration."""

    pass


@runtime_checkable
class E2bSandboxHostClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class E2bSandboxHostIntegration(SandboxHostIntegrationContract):
    """
    E2B sandbox host integration.

    The legacy facade (create_e2b_sandbox_host) remains separate and backward-compatible.
    """

    config: E2bSandboxHostIntegrationConfig = E2bSandboxHostIntegrationConfig()
    _client: E2bSandboxHostClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: E2bSandboxHostClient,
        *,
        enabled: bool = False,
    ) -> E2bSandboxHostIntegration:
        integration = cls.for_provider(
            provider_id=E2B_SANDBOX_HOST_PROVIDER_ID,
            display_name="E2B",
            config=E2bSandboxHostIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> E2bSandboxHostClient | None:
        return self._client
