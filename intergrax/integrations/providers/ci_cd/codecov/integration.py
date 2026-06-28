# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Codecov ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Codecov ci cd integration.

    The legacy facade (create_codecov_ci_cd) remains separate and backward-compatible.
    """

    config: CodecovCiCdIntegrationConfig = CodecovCiCdIntegrationConfig()
    _client: CodecovCiCdClient | None = PrivateAttr(default=None)

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
