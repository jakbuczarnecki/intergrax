# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Airbyte workflow orchestrator integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
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
    Single public Airbyte workflow orchestrator entrypoint.

    Legacy catalog factory (create_airbyte_workflow_orchestrator) delegates to this class.
    """

    config: AirbyteWorkflowOrchestratorIntegrationConfig = AirbyteWorkflowOrchestratorIntegrationConfig()
    _client: AirbyteWorkflowOrchestratorClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> AirbyteWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=AIRBYTE_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="Airbyte",
            config=AirbyteWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Airbyte integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

WorkflowOrchestratorBackend.register(AirbyteWorkflowOrchestratorIntegration)
