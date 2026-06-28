# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_n8n_workflow_orchestrator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.workflow_orchestrator.n8n.integration import (
    N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    N8nWorkflowOrchestratorIntegration,
    N8nWorkflowOrchestratorIntegrationConfig,
    N8nWorkflowOrchestratorClient,
)

__all__ = [
    "create_n8n_workflow_orchestrator",
    "create_n8n_workflow_orchestrator_integration",
]


def create_n8n_workflow_orchestrator_integration(
    *,
    client: N8nWorkflowOrchestratorClient | None = None,
    enabled: bool = False,
) -> N8nWorkflowOrchestratorIntegration:
    """
    Build a contract-based n8n workflow orchestrator integration.

    The legacy facade (create_n8n_workflow_orchestrator) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "n8n workflow orchestrator integration requires an injected client when enabled=True",
        )
    if client is not None:
        return N8nWorkflowOrchestratorIntegration.from_client(client, enabled=enabled)
    return N8nWorkflowOrchestratorIntegration.for_provider(
        provider_id=N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
        display_name="n8n",
        config=N8nWorkflowOrchestratorIntegrationConfig(enabled=enabled),
    )
