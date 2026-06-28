# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prefect workflow orchestrator integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PREFECT_WORKFLOW_ORCHESTRATOR_PROVIDER_ID = "prefect"


class PrefectWorkflowOrchestratorIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Prefect workflow orchestrator integration."""

    pass


@runtime_checkable
class PrefectWorkflowOrchestratorClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PrefectWorkflowOrchestratorIntegration(WorkflowOrchestratorIntegrationContract):
    """
    Prefect workflow orchestrator integration.

    The legacy facade (create_prefect_workflow_orchestrator) remains separate and backward-compatible.
    """

    config: PrefectWorkflowOrchestratorIntegrationConfig = PrefectWorkflowOrchestratorIntegrationConfig()
    _client: PrefectWorkflowOrchestratorClient | None = PrivateAttr(default=None)

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
