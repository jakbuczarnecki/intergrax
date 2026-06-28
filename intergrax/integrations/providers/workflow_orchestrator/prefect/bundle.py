# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_prefect_workflow_orchestrator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.workflow_orchestrator.prefect.integration import (
    PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    PrefectWorkflowOrchestratorIntegration,
    PrefectWorkflowOrchestratorIntegrationConfig,
    PrefectWorkflowOrchestratorClient,
)

__all__ = [
    "create_prefect_workflow_orchestrator",
    "create_prefect_workflow_orchestrator_integration",
]


def create_prefect_workflow_orchestrator_integration(
    *,
    client: PrefectWorkflowOrchestratorClient | None = None,
    enabled: bool = False,
) -> PrefectWorkflowOrchestratorIntegration:
    """
    Build a contract-based Prefect workflow orchestrator integration.

    The legacy facade (create_prefect_workflow_orchestrator) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Prefect workflow orchestrator integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PrefectWorkflowOrchestratorIntegration.from_client(client, enabled=enabled)
    return PrefectWorkflowOrchestratorIntegration.for_provider(
        provider_id=PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
        display_name="Prefect",
        config=PrefectWorkflowOrchestratorIntegrationConfig(enabled=enabled),
    )
