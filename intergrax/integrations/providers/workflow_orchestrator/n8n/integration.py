# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""N8N workflow orchestrator integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID = "n8n"


class N8nWorkflowOrchestratorIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for N8N workflow orchestrator integration."""

    pass


N8nWorkflowOrchestratorClient = WorkflowOrchestratorBackend

class N8nWorkflowOrchestratorIntegration(WorkflowOrchestratorIntegrationContract):
    """
    Single public N8N workflow orchestrator entrypoint.

    Legacy catalog factory (create_n8n_workflow_orchestrator) owns catalog behavior; legacy factories use from_client().
    """

    config: N8nWorkflowOrchestratorIntegrationConfig = N8nWorkflowOrchestratorIntegrationConfig()
    _client: N8nWorkflowOrchestratorClient | None = PrivateAttr(default=None)
    

    def cancel_run(self, run_id):
        return self._require_client().cancel_run(run_id)

    def fetch_logs(self, run_id, tail_lines: int = 200):
        return self._require_client().fetch_logs(run_id, tail_lines=tail_lines)

    def list_runs(self, workflow_id: str = '', limit: int = 20):
        return self._require_client().list_runs(workflow_id=workflow_id, limit=limit)

    def poll_status(self, run_id):
        return self._require_client().poll_status(run_id)

    def trigger_run(self, workflow_id, parameters: dict[str, str] | None = None):
        return self._require_client().trigger_run(workflow_id, parameters=parameters)

    def _require_client(self) -> WorkflowOrchestratorBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: N8nWorkflowOrchestratorClient,
        *,
        enabled: bool = False,
    ) -> N8nWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="N8N",
            config=N8nWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> N8nWorkflowOrchestratorClient | None:
        return self._client

WorkflowOrchestratorBackend.register(N8nWorkflowOrchestratorIntegration)
