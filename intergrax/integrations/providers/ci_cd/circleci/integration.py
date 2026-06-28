# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Circleci ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CIRCLECI_CI_CD_PROVIDER_ID = "circleci"


class CircleciCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Circleci ci cd integration."""

    pass


@runtime_checkable
class CircleciCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CircleciCiCdIntegration(CiCdIntegrationContract):
    """
    Circleci ci cd integration.

    The legacy facade (create_circleci_ci_cd) remains separate and backward-compatible.
    """

    config: CircleciCiCdIntegrationConfig = CircleciCiCdIntegrationConfig()
    _client: CircleciCiCdClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: CircleciCiCdClient,
        *,
        enabled: bool = False,
    ) -> CircleciCiCdIntegration:
        integration = cls.for_provider(
            provider_id=CIRCLECI_CI_CD_PROVIDER_ID,
            display_name="Circleci",
            config=CircleciCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CircleciCiCdClient | None:
        return self._client
