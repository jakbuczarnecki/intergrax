# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Airbyte workflow orchestrator."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.airbyte.bundle import (
    create_airbyte_workflow_orchestrator_integration,
)
from intergrax.integrations.providers.workflow_orchestrator.airbyte.integration import (
    AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    AirbyteWorkflowOrchestratorIntegration,
    AirbyteWorkflowOrchestratorIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="workflow_orchestrator",
    provider_id=AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    integration_class=AirbyteWorkflowOrchestratorIntegration,
    contract_class=WorkflowOrchestratorIntegrationContract,
    contract_factory=create_airbyte_workflow_orchestrator_integration,
    display_name="Airbyte",
    config_class=AirbyteWorkflowOrchestratorIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
