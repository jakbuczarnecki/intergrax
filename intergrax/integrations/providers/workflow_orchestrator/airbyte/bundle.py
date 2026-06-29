# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_airbyte_workflow_orchestrator as _legacy_create_airbyte_workflow_orchestrator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.workflow_orchestrator.airbyte.integration import (
    AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
    AirbyteWorkflowOrchestratorIntegration,
    AirbyteWorkflowOrchestratorIntegrationConfig,
    AirbyteWorkflowOrchestratorClient,
)

__all__ = [
    "create_airbyte_workflow_orchestrator",
    "create_airbyte_workflow_orchestrator_integration",
]


def create_airbyte_workflow_orchestrator_integration(
    *,
    client: AirbyteWorkflowOrchestratorClient | None = None,
    enabled: bool = False,
) -> AirbyteWorkflowOrchestratorIntegration:
    """
    Build a contract-based Airbyte workflow orchestrator integration.

    The legacy facade (create_airbyte_workflow_orchestrator) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Airbyte workflow orchestrator integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AirbyteWorkflowOrchestratorIntegration.from_client(client, enabled=enabled)
    return AirbyteWorkflowOrchestratorIntegration.for_provider(
        provider_id=AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
        display_name="Airbyte",
        config=AirbyteWorkflowOrchestratorIntegrationConfig(enabled=enabled),
    )


def create_airbyte_workflow_orchestrator(**kwargs: object) -> AirbyteWorkflowOrchestratorIntegration:
    """Compatibility shim — constructs AirbyteWorkflowOrchestratorIntegration from legacy runtime."""
    runtime = _legacy_create_airbyte_workflow_orchestrator(**kwargs)
    if isinstance(runtime, AirbyteWorkflowOrchestratorIntegration):
        return runtime
    return AirbyteWorkflowOrchestratorIntegration.from_runtime(runtime)
