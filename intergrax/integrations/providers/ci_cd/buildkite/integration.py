# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Buildkite ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BUILDKITE_CI_CD_PROVIDER_ID = "buildkite"


class BuildkiteCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Buildkite ci cd integration."""

    pass


@runtime_checkable
class BuildkiteCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BuildkiteCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Buildkite ci cd entrypoint.

    Legacy catalog factory (create_buildkite_ci_cd) delegates to this class.
    """

    config: BuildkiteCiCdIntegrationConfig = BuildkiteCiCdIntegrationConfig()
    _client: BuildkiteCiCdClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> BuildkiteCiCdIntegration:
        integration = cls.for_provider(
            provider_id=BUILDKITE_CI_CD_PROVIDER_ID,
            display_name="Buildkite",
            config=BuildkiteCiCdIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Buildkite integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: BuildkiteCiCdClient,
        *,
        enabled: bool = False,
    ) -> BuildkiteCiCdIntegration:
        integration = cls.for_provider(
            provider_id=BUILDKITE_CI_CD_PROVIDER_ID,
            display_name="Buildkite",
            config=BuildkiteCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BuildkiteCiCdClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CiCdBackend.register(BuildkiteCiCdIntegration)
