"""Google Workspace Vendor Knowledge contribution."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationDependencyError
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceClientFactory,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.tenant_connection_factory import (
    GoogleWorkspaceTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    register_google_workspace_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_docs import (
    register_google_workspace_docs_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_drive import (
    register_google_workspace_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_sheets import (
    register_google_workspace_sheets_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import (
    build_adapter,
    build_durable_source_plugin,
)


class _UnavailableGoogleWorkspaceClientFactory:
    """Keep the canonical route present until an auth executor is composed."""

    def create_client_family(self, *, credential_material: object) -> object:
        raise IntegrationDependencyError(
            "Google Workspace client family is unavailable in this composition",
        )


def build_google_workspace_vendor_knowledge_contribution(
    *,
    client_factory: GoogleWorkspaceClientFactory | None = None,
) -> VendorKnowledgeProviderContribution:
    provider_id = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    category = IntegrationCategory.COLLABORATION_SUITE
    return VendorKnowledgeProviderContribution(
        provider_id=provider_id,
        integration_category=category,
        adapters=(
            build_adapter(register_google_workspace_drive_knowledge_adapter),
            build_adapter(register_google_workspace_docs_knowledge_adapter),
            build_adapter(register_google_workspace_sheets_knowledge_adapter),
            build_adapter(register_google_workspace_calendar_knowledge_adapter),
        ),
        source_plugins=(
            build_durable_source_plugin(
                provider_id=provider_id,
                integration_category=category,
                source_kind=GOOGLE_DRIVE_SOURCE_KIND,
                runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:drive",
            ),
            build_durable_source_plugin(
                provider_id=provider_id,
                integration_category=category,
                source_kind=GOOGLE_DOCS_SOURCE_KIND,
                runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:docs",
                indexed_runtime_ref="indexed-source:google_workspace:docs",
            ),
            build_durable_source_plugin(
                provider_id=provider_id,
                integration_category=category,
                source_kind=GOOGLE_SHEETS_SOURCE_KIND,
                runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:sheets",
                indexed_runtime_ref="indexed-source:google_workspace:sheets",
            ),
            build_durable_source_plugin(
                provider_id=provider_id,
                integration_category=category,
                source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
                runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:calendar",
                indexed_runtime_ref="indexed-source:google_workspace:calendar",
            ),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=provider_id,
                integration_category=category,
                factory=GoogleWorkspaceTenantConnectionIntegrationFactory(
                    client_factory=client_factory
                    or _UnavailableGoogleWorkspaceClientFactory(),
                ),
            ),
        ),
    )


__all__ = ["build_google_workspace_vendor_knowledge_contribution"]
