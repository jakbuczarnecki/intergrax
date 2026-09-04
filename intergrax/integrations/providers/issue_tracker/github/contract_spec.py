# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Github issue tracker."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.github.bundle import (
    create_github_issue_tracker_integration,
)
from intergrax.integrations.providers.issue_tracker.github.integration import (
    GITHUB_ISSUE_TRACKER_PROVIDER_ID,
    GithubIssueTrackerIntegration,
    GithubIssueTrackerIntegrationConfig,
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
    provider_id=GITHUB_ISSUE_TRACKER_PROVIDER_ID,
    integration_class=GithubIssueTrackerIntegration,
    contract_class=IssueTrackerIntegrationContract,
    contract_factory=create_github_issue_tracker_integration,
    display_name="Github",
    config_class=GithubIssueTrackerIntegrationConfig,
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
