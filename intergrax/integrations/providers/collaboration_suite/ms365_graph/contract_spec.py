# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Ms365 Graph collaboration suite."""

from __future__ import annotations

from intergrax.integrations.providers.collaboration_suite.ms365_graph.bundle import (
    create_ms365_graph_collaboration_suite_integration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
    Ms365GraphCollaborationSuiteIntegrationConfig,
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
    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_class=Ms365GraphCollaborationSuiteIntegration,
    contract_class=CollaborationSuiteIntegrationContract,
    contract_factory=create_ms365_graph_collaboration_suite_integration,
    display_name="Ms365 Graph",
    config_class=Ms365GraphCollaborationSuiteIntegrationConfig,
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
