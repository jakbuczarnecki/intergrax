# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Airflow workflow orchestrator."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.airflow.bundle import (
    create_airflow_workflow_orchestrator_integration,
)
from intergrax.integrations.providers.workflow_orchestrator.airflow.integration import (
    AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    AirflowWorkflowOrchestratorIntegration,
    AirflowWorkflowOrchestratorIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="workflow_orchestrator",
    provider_id=AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    integration_class=AirflowWorkflowOrchestratorIntegration,
    contract_class=WorkflowOrchestratorIntegrationContract,
    contract_factory=create_airflow_workflow_orchestrator_integration,
    display_name="Airflow",
    config_class=AirflowWorkflowOrchestratorIntegrationConfig,
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
