# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Prefect workflow orchestrator."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.prefect.bundle import (
    create_prefect_workflow_orchestrator_integration,
)
from intergrax.integrations.providers.workflow_orchestrator.prefect.integration import (
    PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    PrefectWorkflowOrchestratorIntegration,
    PrefectWorkflowOrchestratorIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="workflow_orchestrator",
    provider_id=PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    integration_class=PrefectWorkflowOrchestratorIntegration,
    contract_class=WorkflowOrchestratorIntegrationContract,
    contract_factory=create_prefect_workflow_orchestrator_integration,
    display_name="Prefect",
    config_class=PrefectWorkflowOrchestratorIntegrationConfig,
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
