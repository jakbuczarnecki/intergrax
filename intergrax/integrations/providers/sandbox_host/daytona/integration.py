# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Daytona sandbox host integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
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
    Single public Daytona sandbox host entrypoint.

    Legacy catalog factory (create_daytona_sandbox_host) delegates to this class.
    """

    config: DaytonaSandboxHostIntegrationConfig = DaytonaSandboxHostIntegrationConfig()
    _client: DaytonaSandboxHostClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> DaytonaSandboxHostIntegration:
        integration = cls.for_provider(
            provider_id=DAYTONA_SANDBOX_HOST_PROVIDER_ID,
            display_name="Daytona",
            config=DaytonaSandboxHostIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Daytona integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SandboxHostBackend.register(DaytonaSandboxHostIntegration)
