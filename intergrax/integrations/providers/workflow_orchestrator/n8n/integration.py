# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""n8n workflow orchestrator integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID = "n8n"


class N8nWorkflowOrchestratorIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for n8n workflow orchestrator integration."""

    pass


@runtime_checkable
class N8nWorkflowOrchestratorClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class N8nWorkflowOrchestratorIntegration(WorkflowOrchestratorIntegrationContract):
    """
    n8n workflow orchestrator integration.

    The legacy facade (create_n8n_workflow_orchestrator) remains separate and backward-compatible.
    """

    config: N8nWorkflowOrchestratorIntegrationConfig = N8nWorkflowOrchestratorIntegrationConfig()
    _client: N8nWorkflowOrchestratorClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: N8nWorkflowOrchestratorClient,
        *,
        enabled: bool = False,
    ) -> N8nWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=N8N_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="n8n",
            config=N8nWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> N8nWorkflowOrchestratorClient | None:
        return self._client
