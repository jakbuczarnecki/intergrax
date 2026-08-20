# © Artur Czarnecki. All rights reserved.

"""Governance Approval Vendor Knowledge contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.governance_approval.config import (
    GovernanceApprovalIntegrationConfig,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
)
from intergrax.integrations.providers.governance_approval.tenant_connection_factory import (
    GovernanceApprovalTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.live.governance_approval.registration import (
    build_governance_approval_live_registration_bundles,
    build_governance_approval_vendor_knowledge_source_plugin,
)


def build_governance_approval_vendor_knowledge_contribution(
    *,
    http_client_factory: Callable[[GovernanceApprovalIntegrationConfig], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.WORKFLOW_ORCHESTRATOR
    return VendorKnowledgeProviderContribution(
        provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
        integration_category=category,
        source_plugins=(build_governance_approval_vendor_knowledge_source_plugin(),),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
                integration_category=category,
                factory=GovernanceApprovalTenantConnectionIntegrationFactory(
                    http_client_factory=http_client_factory,
                ),
            ),
        ),
        live_contributions=build_governance_approval_live_registration_bundles(),
    )


__all__ = ["build_governance_approval_vendor_knowledge_contribution"]
