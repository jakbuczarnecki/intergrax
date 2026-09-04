# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Google Workspace collaboration suite."""

from __future__ import annotations

from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import (
    create_google_workspace_collaboration_suite_integration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.collaboration import (
    CollaborationSuiteIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="collaboration_suite",
    provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_class=GoogleWorkspaceCollaborationSuiteIntegration,
    contract_class=CollaborationSuiteIntegrationContract,
    contract_factory=create_google_workspace_collaboration_suite_integration,
    display_name="Google Workspace",
    config_class=GoogleWorkspaceCollaborationSuiteIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={
        "source": "explicit_provider_declaration"
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
