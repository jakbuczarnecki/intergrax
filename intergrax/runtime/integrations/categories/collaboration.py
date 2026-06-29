# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Collaboration and knowledge provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_HEALTH,
    _CONNECT_READ_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

COLLABORATION_SUITE_INTEGRATION_CONTRACT_SCHEMA = "collaboration_suite_integration_contract.v1"
ISSUE_TRACKER_INTEGRATION_CONTRACT_SCHEMA = "issue_tracker_integration_contract.v1"
WIKI_KNOWLEDGE_INTEGRATION_CONTRACT_SCHEMA = "wiki_knowledge_integration_contract.v1"
INTERACTION_SURFACE_INTEGRATION_CONTRACT_SCHEMA = "interaction_surface_integration_contract.v1"


class CollaborationSuiteIntegrationContract(PlatformIntegrationContract):
    """Category contract for collaboration_suite providers (ms365_graph, google_workspace, …)."""

    schema_id: Literal["collaboration_suite_integration_contract.v1"] = (
        COLLABORATION_SUITE_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.COLLABORATION_SUITE.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> CollaborationSuiteIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.COLLABORATION_SUITE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class IssueTrackerIntegrationContract(PlatformIntegrationContract):
    """Category contract for issue_tracker providers (jira, github, linear, …)."""

    schema_id: Literal["issue_tracker_integration_contract.v1"] = ISSUE_TRACKER_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.ISSUE_TRACKER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> IssueTrackerIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.ISSUE_TRACKER.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class WikiKnowledgeIntegrationContract(PlatformIntegrationContract):
    """Category contract for wiki_knowledge providers (confluence, notion, …)."""

    schema_id: Literal["wiki_knowledge_integration_contract.v1"] = WIKI_KNOWLEDGE_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.WIKI_KNOWLEDGE.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> WikiKnowledgeIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.WIKI_KNOWLEDGE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class InteractionSurfaceIntegrationContract(PlatformIntegrationContract):
    """Category contract for interaction_surface providers (lab_json, slash_command, …)."""

    schema_id: Literal["interaction_surface_integration_contract.v1"] = (
        INTERACTION_SURFACE_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.INTERACTION_SURFACE.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> InteractionSurfaceIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.INTERACTION_SURFACE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )
