# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.change_approval.config import ChangeApprovalIntegrationConfig
from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
    CHANGE_APPROVAL_SOURCE_KIND,
    ChangeApprovalReadClient,
    ChangeApprovalSnapshotV1,
)
from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract

__all__ = [
    "CHANGE_APPROVAL_PROVIDER_ID",
    "CHANGE_APPROVAL_SOURCE_KIND",
    "ChangeApprovalIntegration",
    "ChangeApprovalIntegrationConfig",
]


class ChangeApprovalIntegration(IssueTrackerIntegrationContract):
    """Single public Change Approval entrypoint for Vendor Knowledge live reads."""

    config: ChangeApprovalIntegrationConfig = ChangeApprovalIntegrationConfig(
        base_url="http://127.0.0.1:8767",
    )
    _client: ChangeApprovalReadClient | None = PrivateAttr(default=None)

    async def read_change_approval(self, *, change_id: str) -> ChangeApprovalSnapshotV1:
        return await self._require_client().read_change_approval(change_id=change_id)

    def _require_client(self) -> ChangeApprovalReadClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a configured read client",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: ChangeApprovalReadClient,
        *,
        config: ChangeApprovalIntegrationConfig | None = None,
    ) -> ChangeApprovalIntegration:
        integration = cls.for_provider(
            provider_id=CHANGE_APPROVAL_PROVIDER_ID,
            display_name="Change Approval",
            config=config
            or ChangeApprovalIntegrationConfig(
                base_url="http://127.0.0.1:8767",
            ),
        )
        integration._client = client
        return integration
