# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Argocd ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ARGOCD_CI_CD_PROVIDER_ID = "argocd"


class ArgocdCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Argocd ci cd integration."""

    pass


@runtime_checkable
class ArgocdCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ArgocdCiCdIntegration(CiCdIntegrationContract):
    """
    Argocd ci cd integration.

    The legacy facade (create_argocd_ci_cd) remains separate and backward-compatible.
    """

    config: ArgocdCiCdIntegrationConfig = ArgocdCiCdIntegrationConfig()
    _client: ArgocdCiCdClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ArgocdCiCdClient,
        *,
        enabled: bool = False,
    ) -> ArgocdCiCdIntegration:
        integration = cls.for_provider(
            provider_id=ARGOCD_CI_CD_PROVIDER_ID,
            display_name="Argocd",
            config=ArgocdCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ArgocdCiCdClient | None:
        return self._client
