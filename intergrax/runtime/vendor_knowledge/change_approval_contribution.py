# © Artur Czarnecki. All rights reserved.

"""Change Approval Vendor Knowledge contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.change_approval.config import ChangeApprovalIntegrationConfig
from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
)
from intergrax.integrations.providers.change_approval.tenant_connection_factory import (
    ChangeApprovalTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.live.change_approval.registration import (
    build_change_approval_live_registration_bundles,
    build_change_approval_vendor_knowledge_source_plugin,
)


def build_change_approval_vendor_knowledge_contribution(
    *,
    http_client_factory: Callable[[ChangeApprovalIntegrationConfig], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.ISSUE_TRACKER
    return VendorKnowledgeProviderContribution(
        provider_id=CHANGE_APPROVAL_PROVIDER_ID,
        integration_category=category,
        source_plugins=(build_change_approval_vendor_knowledge_source_plugin(),),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=CHANGE_APPROVAL_PROVIDER_ID,
                integration_category=category,
                factory=ChangeApprovalTenantConnectionIntegrationFactory(
                    http_client_factory=http_client_factory,
                ),
            ),
        ),
        live_contributions=build_change_approval_live_registration_bundles(),
    )


__all__ = ["build_change_approval_vendor_knowledge_contribution"]
