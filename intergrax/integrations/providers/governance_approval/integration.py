# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.governance_approval.config import (
    GovernanceApprovalIntegrationConfig,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
    GOVERNANCE_APPROVAL_SOURCE_KIND,
    GovernanceApprovalReadClient,
    GovernanceApprovalSnapshotV1,
)
from intergrax.runtime.integrations.categories.devops import (
    WorkflowOrchestratorIntegrationContract,
)

__all__ = [
    "GOVERNANCE_APPROVAL_PROVIDER_ID",
    "GOVERNANCE_APPROVAL_SOURCE_KIND",
    "GovernanceApprovalIntegration",
    "GovernanceApprovalIntegrationConfig",
]


class GovernanceApprovalIntegration(WorkflowOrchestratorIntegrationContract):
    """Single public Governance Approval entrypoint for Vendor Knowledge live reads."""

    config: GovernanceApprovalIntegrationConfig = GovernanceApprovalIntegrationConfig(
        base_url="http://127.0.0.1:8768",
    )
    _client: GovernanceApprovalReadClient | None = PrivateAttr(default=None)

    async def read_governance_approval(
        self,
        *,
        subject_id: str,
    ) -> GovernanceApprovalSnapshotV1:
        return await self._require_client().read_governance_approval(subject_id=subject_id)

    def _require_client(self) -> GovernanceApprovalReadClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a configured read client",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: GovernanceApprovalReadClient,
        *,
        config: GovernanceApprovalIntegrationConfig | None = None,
    ) -> GovernanceApprovalIntegration:
        integration = cls.for_provider(
            provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
            display_name="Governance Approval",
            config=config
            or GovernanceApprovalIntegrationConfig(
                base_url="http://127.0.0.1:8768",
            ),
        )
        integration._client = client
        return integration
