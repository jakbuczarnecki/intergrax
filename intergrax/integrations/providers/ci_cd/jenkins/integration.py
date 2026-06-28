# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jenkins ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Jenkins ci cd integration.

    The legacy facade (create_jenkins_ci_cd) remains separate and backward-compatible.
    """

    config: JenkinsCiCdIntegrationConfig = JenkinsCiCdIntegrationConfig()
    _client: JenkinsCiCdClient | None = PrivateAttr(default=None)

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
