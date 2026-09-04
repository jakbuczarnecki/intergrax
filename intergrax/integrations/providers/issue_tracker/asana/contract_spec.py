# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Asana issue tracker."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.asana.bundle import (
    create_asana_issue_tracker_integration,
)
from intergrax.integrations.providers.issue_tracker.asana.integration import (
    ASANA_ISSUE_TRACKER_PROVIDER_ID,
    AsanaIssueTrackerIntegration,
    AsanaIssueTrackerIntegrationConfig,
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
    provider_id=ASANA_ISSUE_TRACKER_PROVIDER_ID,
    integration_class=AsanaIssueTrackerIntegration,
    contract_class=IssueTrackerIntegrationContract,
    contract_factory=create_asana_issue_tracker_integration,
    display_name="Asana",
    config_class=AsanaIssueTrackerIntegrationConfig,
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
