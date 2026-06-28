# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vault secrets store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

VAULT_SECRETS_STORE_PROVIDER_ID = "vault"


class VaultSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Vault secrets store integration."""

    pass


@runtime_checkable
class VaultSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class VaultSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Vault secrets store integration.

    The legacy facade (create_vault_secrets_store) remains separate and backward-compatible.
    """

    config: VaultSecretsStoreIntegrationConfig = VaultSecretsStoreIntegrationConfig()
    _client: VaultSecretsStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: VaultSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> VaultSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=VAULT_SECRETS_STORE_PROVIDER_ID,
            display_name="Vault",
            config=VaultSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> VaultSecretsStoreClient | None:
        return self._client
