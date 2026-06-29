# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jenkins ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

JENKINS_CI_CD_PROVIDER_ID = "jenkins"


class JenkinsCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Jenkins ci cd integration."""

    pass


@runtime_checkable
class JenkinsCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class JenkinsCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Jenkins ci cd entrypoint.

    Legacy catalog factory (create_jenkins_ci_cd) delegates to this class.
    """

    config: JenkinsCiCdIntegrationConfig = JenkinsCiCdIntegrationConfig()
    _client: JenkinsCiCdClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> JenkinsCiCdIntegration:
        integration = cls.for_provider(
            provider_id=JENKINS_CI_CD_PROVIDER_ID,
            display_name="Jenkins",
            config=JenkinsCiCdIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Jenkins integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: JenkinsCiCdClient,
        *,
        enabled: bool = False,
    ) -> JenkinsCiCdIntegration:
        integration = cls.for_provider(
            provider_id=JENKINS_CI_CD_PROVIDER_ID,
            display_name="Jenkins",
            config=JenkinsCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> JenkinsCiCdClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CiCdBackend.register(JenkinsCiCdIntegration)
