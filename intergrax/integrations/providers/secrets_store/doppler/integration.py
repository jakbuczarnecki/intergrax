# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Doppler secrets store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DOPPLER_SECRETS_STORE_PROVIDER_ID = "doppler"


class DopplerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Doppler secrets store integration."""

    pass


@runtime_checkable
class DopplerSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DopplerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Doppler secrets store integration.

    The legacy facade (create_doppler_secrets_store) remains separate and backward-compatible.
    """

    config: DopplerSecretsStoreIntegrationConfig = DopplerSecretsStoreIntegrationConfig()
    _client: DopplerSecretsStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: DopplerSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> DopplerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=DOPPLER_SECRETS_STORE_PROVIDER_ID,
            display_name="Doppler",
            config=DopplerSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DopplerSecretsStoreClient | None:
        return self._client
