# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Github Actions ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GITHUB_ACTIONS_CI_CD_PROVIDER_ID = "github_actions"


class GithubActionsCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Github Actions ci cd integration."""

    pass


@runtime_checkable
class GithubActionsCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GithubActionsCiCdIntegration(CiCdIntegrationContract):
    """
    Github Actions ci cd integration.

    The legacy facade (create_github_actions_ci_cd) remains separate and backward-compatible.
    """

    config: GithubActionsCiCdIntegrationConfig = GithubActionsCiCdIntegrationConfig()
    _client: GithubActionsCiCdClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GithubActionsCiCdClient,
        *,
        enabled: bool = False,
    ) -> GithubActionsCiCdIntegration:
        integration = cls.for_provider(
            provider_id=GITHUB_ACTIONS_CI_CD_PROVIDER_ID,
            display_name="Github Actions",
            config=GithubActionsCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GithubActionsCiCdClient | None:
        return self._client
