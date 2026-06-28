# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Buildkite ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Buildkite ci cd integration.

    The legacy facade (create_buildkite_ci_cd) remains separate and backward-compatible.
    """

    config: BuildkiteCiCdIntegrationConfig = BuildkiteCiCdIntegrationConfig()
    _client: BuildkiteCiCdClient | None = PrivateAttr(default=None)

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
