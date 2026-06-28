# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Daytona sandbox host integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DAYTONA_SANDBOX_HOST_PROVIDER_ID = "daytona"


class DaytonaSandboxHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Daytona sandbox host integration."""

    pass


@runtime_checkable
class DaytonaSandboxHostClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DaytonaSandboxHostIntegration(SandboxHostIntegrationContract):
    """
    Daytona sandbox host integration.

    The legacy facade (create_daytona_sandbox_host) remains separate and backward-compatible.
    """

    config: DaytonaSandboxHostIntegrationConfig = DaytonaSandboxHostIntegrationConfig()
    _client: DaytonaSandboxHostClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: DaytonaSandboxHostClient,
        *,
        enabled: bool = False,
    ) -> DaytonaSandboxHostIntegration:
        integration = cls.for_provider(
            provider_id=DAYTONA_SANDBOX_HOST_PROVIDER_ID,
            display_name="Daytona",
            config=DaytonaSandboxHostIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DaytonaSandboxHostClient | None:
        return self._client
