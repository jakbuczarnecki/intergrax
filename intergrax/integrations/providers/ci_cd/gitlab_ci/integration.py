# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gitlab Ci ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GITLAB_CI_CI_CD_PROVIDER_ID = "gitlab_ci"


class GitlabCiCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gitlab Ci ci cd integration."""

    pass


@runtime_checkable
class GitlabCiCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GitlabCiCiCdIntegration(CiCdIntegrationContract):
    """
    Gitlab Ci ci cd integration.

    The legacy facade (create_gitlab_ci_ci_cd) remains separate and backward-compatible.
    """

    config: GitlabCiCiCdIntegrationConfig = GitlabCiCiCdIntegrationConfig()
    _client: GitlabCiCiCdClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GitlabCiCiCdClient,
        *,
        enabled: bool = False,
    ) -> GitlabCiCiCdIntegration:
        integration = cls.for_provider(
            provider_id=GITLAB_CI_CI_CD_PROVIDER_ID,
            display_name="Gitlab Ci",
            config=GitlabCiCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GitlabCiCiCdClient | None:
        return self._client
