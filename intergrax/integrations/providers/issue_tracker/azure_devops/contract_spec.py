# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Azure Devops issue tracker."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.azure_devops.bundle import (
    create_azure_devops_issue_tracker_integration,
)
from intergrax.integrations.providers.issue_tracker.azure_devops.integration import (
    AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
    AzureDevopsIssueTrackerIntegration,
    AzureDevopsIssueTrackerIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.collaboration import (
    IssueTrackerIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="issue_tracker",
    provider_id=AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
    integration_class=AzureDevopsIssueTrackerIntegration,
    contract_class=IssueTrackerIntegrationContract,
    contract_factory=create_azure_devops_issue_tracker_integration,
    display_name="Azure Devops",
    config_class=AzureDevopsIssueTrackerIntegrationConfig,
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
