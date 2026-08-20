# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.security_status.config import SecurityStatusIntegrationConfig
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
    SECURITY_STATUS_SOURCE_KIND,
    SecurityStatusReadClient,
    SecurityStatusSnapshotV1,
)
from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract

__all__ = [
    "SECURITY_STATUS_PROVIDER_ID",
    "SECURITY_STATUS_SOURCE_KIND",
    "SecurityStatusIntegration",
    "SecurityStatusIntegrationConfig",
]


class SecurityStatusIntegration(SecurityScannerIntegrationContract):
    """Single public Security Status entrypoint for Vendor Knowledge live reads."""

    config: SecurityStatusIntegrationConfig = SecurityStatusIntegrationConfig(
        base_url="http://127.0.0.1:8766",
    )
    _client: SecurityStatusReadClient | None = PrivateAttr(default=None)

    async def read_security_status(self, *, project_id: str) -> SecurityStatusSnapshotV1:
        return await self._require_client().read_security_status(project_id=project_id)

    def _require_client(self) -> SecurityStatusReadClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a configured read client",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: SecurityStatusReadClient,
        *,
        config: SecurityStatusIntegrationConfig | None = None,
    ) -> SecurityStatusIntegration:
        integration = cls.for_provider(
            provider_id=SECURITY_STATUS_PROVIDER_ID,
            display_name="Security Status",
            config=config
            or SecurityStatusIntegrationConfig(
                base_url="http://127.0.0.1:8766",
            ),
        )
        integration._client = client
        return integration
