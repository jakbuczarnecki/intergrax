# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Workos identity provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WORKOS_IDENTITY_PROVIDER_PROVIDER_ID = "workos"


class WorkosIdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Workos identity provider integration."""

    pass


@runtime_checkable
class WorkosIdentityProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class WorkosIdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Single public Workos identity provider entrypoint.

    Legacy catalog factory (create_workos_identity_provider) delegates to this class.
    """

    config: WorkosIdentityProviderIntegrationConfig = WorkosIdentityProviderIntegrationConfig()
    _client: WorkosIdentityProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> WorkosIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Workos",
            config=WorkosIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Workos integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: WorkosIdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> WorkosIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Workos",
            config=WorkosIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WorkosIdentityProviderClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

IdentityProviderBackend.register(WorkosIdentityProviderIntegration)
