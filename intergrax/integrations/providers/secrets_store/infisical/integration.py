# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Infisical secrets store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

INFISICAL_SECRETS_STORE_PROVIDER_ID = "infisical"


class InfisicalSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Infisical secrets store integration."""

    pass


@runtime_checkable
class InfisicalSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class InfisicalSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Infisical secrets store integration.

    The legacy facade (create_infisical_secrets_store) remains separate and backward-compatible.
    """

    config: InfisicalSecretsStoreIntegrationConfig = InfisicalSecretsStoreIntegrationConfig()
    _client: InfisicalSecretsStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: InfisicalSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> InfisicalSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=INFISICAL_SECRETS_STORE_PROVIDER_ID,
            display_name="Infisical",
            config=InfisicalSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> InfisicalSecretsStoreClient | None:
        return self._client
