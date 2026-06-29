# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Codecov ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CODECOV_CI_CD_PROVIDER_ID = "codecov"


class CodecovCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Codecov ci cd integration."""

    pass


@runtime_checkable
class CodecovCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CodecovCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Codecov ci cd entrypoint.

    Legacy catalog factory (create_codecov_ci_cd) delegates to this class.
    """

    config: CodecovCiCdIntegrationConfig = CodecovCiCdIntegrationConfig()
    _client: CodecovCiCdClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> CodecovCiCdIntegration:
        integration = cls.for_provider(
            provider_id=CODECOV_CI_CD_PROVIDER_ID,
            display_name="Codecov",
            config=CodecovCiCdIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Codecov integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: CodecovCiCdClient,
        *,
        enabled: bool = False,
    ) -> CodecovCiCdIntegration:
        integration = cls.for_provider(
            provider_id=CODECOV_CI_CD_PROVIDER_ID,
            display_name="Codecov",
            config=CodecovCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CodecovCiCdClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CiCdBackend.register(CodecovCiCdIntegration)
