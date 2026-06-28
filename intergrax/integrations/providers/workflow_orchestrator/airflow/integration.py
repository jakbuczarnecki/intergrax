# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Airflow workflow orchestrator integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID = "airflow"


class AirflowWorkflowOrchestratorIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Airflow workflow orchestrator integration."""

    pass


@runtime_checkable
class AirflowWorkflowOrchestratorClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AirflowWorkflowOrchestratorIntegration(WorkflowOrchestratorIntegrationContract):
    """
    Airflow workflow orchestrator integration.

    The legacy facade (create_airflow_workflow_orchestrator) remains separate and backward-compatible.
    """

    config: AirflowWorkflowOrchestratorIntegrationConfig = AirflowWorkflowOrchestratorIntegrationConfig()
    _client: AirflowWorkflowOrchestratorClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AirflowWorkflowOrchestratorClient,
        *,
        enabled: bool = False,
    ) -> AirflowWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="Airflow",
            config=AirflowWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AirflowWorkflowOrchestratorClient | None:
        return self._client
