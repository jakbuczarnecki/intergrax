# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcp Secret Manager secrets store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID = "gcp_secret_manager"


class GcpSecretManagerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcp Secret Manager secrets store integration."""

    pass


@runtime_checkable
class GcpSecretManagerSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GcpSecretManagerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Gcp Secret Manager secrets store integration.

    The legacy facade (create_gcp_secret_manager_secrets_store) remains separate and backward-compatible.
    """

    config: GcpSecretManagerSecretsStoreIntegrationConfig = GcpSecretManagerSecretsStoreIntegrationConfig()
    _client: GcpSecretManagerSecretsStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GcpSecretManagerSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> GcpSecretManagerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
            display_name="Gcp Secret Manager",
            config=GcpSecretManagerSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcpSecretManagerSecretsStoreClient | None:
        return self._client
