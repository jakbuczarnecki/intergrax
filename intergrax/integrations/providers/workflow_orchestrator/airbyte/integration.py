# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Airbyte workflow orchestrator integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import WorkflowOrchestratorIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID = "airbyte"


class AirbyteWorkflowOrchestratorIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Airbyte workflow orchestrator integration."""

    pass


@runtime_checkable
class AirbyteWorkflowOrchestratorClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AirbyteWorkflowOrchestratorIntegration(WorkflowOrchestratorIntegrationContract):
    """
    Airbyte workflow orchestrator integration.

    The legacy facade (create_airbyte_workflow_orchestrator) remains separate and backward-compatible.
    """

    config: AirbyteWorkflowOrchestratorIntegrationConfig = AirbyteWorkflowOrchestratorIntegrationConfig()
    _client: AirbyteWorkflowOrchestratorClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AirbyteWorkflowOrchestratorClient,
        *,
        enabled: bool = False,
    ) -> AirbyteWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="Airbyte",
            config=AirbyteWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AirbyteWorkflowOrchestratorClient | None:
        return self._client
