# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Aws Secrets Manager secrets store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID = "aws_secrets_manager"


class AwsSecretsManagerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Aws Secrets Manager secrets store integration."""

    pass


@runtime_checkable
class AwsSecretsManagerSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AwsSecretsManagerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Aws Secrets Manager secrets store integration.

    The legacy facade (create_aws_secrets_manager_secrets_store) remains separate and backward-compatible.
    """

    config: AwsSecretsManagerSecretsStoreIntegrationConfig = AwsSecretsManagerSecretsStoreIntegrationConfig()
    _client: AwsSecretsManagerSecretsStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AwsSecretsManagerSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> AwsSecretsManagerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID,
            display_name="Aws Secrets Manager",
            config=AwsSecretsManagerSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AwsSecretsManagerSecretsStoreClient | None:
        return self._client
