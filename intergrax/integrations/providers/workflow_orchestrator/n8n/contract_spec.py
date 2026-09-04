# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for n8n workflow orchestrator."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.n8n.bundle import (
    create_n8n_workflow_orchestrator_integration,
)
from intergrax.integrations.providers.workflow_orchestrator.n8n.integration import (
    N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    N8nWorkflowOrchestratorIntegration,
    N8nWorkflowOrchestratorIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="workflow_orchestrator",
    provider_id=N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    integration_class=N8nWorkflowOrchestratorIntegration,
    contract_class=WorkflowOrchestratorIntegrationContract,
    contract_factory=create_n8n_workflow_orchestrator_integration,
    display_name="n8n",
    config_class=N8nWorkflowOrchestratorIntegrationConfig,
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
