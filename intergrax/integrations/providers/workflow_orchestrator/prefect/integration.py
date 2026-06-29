# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prefect workflow orchestrator integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID = "prefect"


class PrefectWorkflowOrchestratorIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Prefect workflow orchestrator integration."""

    pass


PrefectWorkflowOrchestratorClient = WorkflowOrchestratorBackend

class PrefectWorkflowOrchestratorIntegration(WorkflowOrchestratorIntegrationContract):
    """
    Single public Prefect workflow orchestrator entrypoint.

    Legacy catalog factory (create_prefect_workflow_orchestrator) owns catalog behavior; legacy factories use from_client().
    """

    config: PrefectWorkflowOrchestratorIntegrationConfig = PrefectWorkflowOrchestratorIntegrationConfig()
    _client: PrefectWorkflowOrchestratorClient | None = PrivateAttr(default=None)
    

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
        client: PrefectWorkflowOrchestratorClient,
        *,
        enabled: bool = False,
    ) -> PrefectWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="Prefect",
            config=PrefectWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PrefectWorkflowOrchestratorClient | None:
        return self._client

WorkflowOrchestratorBackend.register(PrefectWorkflowOrchestratorIntegration)
