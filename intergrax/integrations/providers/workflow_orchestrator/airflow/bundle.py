# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_airflow_workflow_orchestrator as _legacy_create_airflow_workflow_orchestrator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.workflow_orchestrator.airflow.integration import (
    AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    AirflowWorkflowOrchestratorIntegration,
    AirflowWorkflowOrchestratorIntegrationConfig,
    AirflowWorkflowOrchestratorClient,
)

__all__ = [
    "create_airflow_workflow_orchestrator",
    "create_airflow_workflow_orchestrator_integration",
]


def create_airflow_workflow_orchestrator_integration(
    *,
    client: AirflowWorkflowOrchestratorClient | None = None,
    enabled: bool = False,
) -> AirflowWorkflowOrchestratorIntegration:
    """
    Build a contract-based Airflow workflow orchestrator integration.

    The legacy facade (create_airflow_workflow_orchestrator) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Airflow workflow orchestrator integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AirflowWorkflowOrchestratorIntegration.from_client(client, enabled=enabled)
    return AirflowWorkflowOrchestratorIntegration.for_provider(
        provider_id=AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
        display_name="Airflow",
        config=AirflowWorkflowOrchestratorIntegrationConfig(enabled=enabled),
    )


def create_airflow_workflow_orchestrator(**kwargs: object) -> AirflowWorkflowOrchestratorIntegration:
    """Compatibility shim — constructs AirflowWorkflowOrchestratorIntegration from legacy runtime."""
    runtime = _legacy_create_airflow_workflow_orchestrator(**kwargs)
    if isinstance(runtime, AirflowWorkflowOrchestratorIntegration):
        return runtime
    return AirflowWorkflowOrchestratorIntegration.from_client(runtime)
