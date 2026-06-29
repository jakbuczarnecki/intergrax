# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Airflow workflow orchestrator integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
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
    Single public Airflow workflow orchestrator entrypoint.

    Legacy catalog factory (create_airflow_workflow_orchestrator) delegates to this class.
    """

    config: AirflowWorkflowOrchestratorIntegrationConfig = AirflowWorkflowOrchestratorIntegrationConfig()
    _client: AirflowWorkflowOrchestratorClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> AirflowWorkflowOrchestratorIntegration:
        integration = cls.for_provider(
            provider_id=AIRFLOW_WORKFLOW_ORCHESTRATOR_PROVIDER_ID,
            display_name="Airflow",
            config=AirflowWorkflowOrchestratorIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Airflow integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

WorkflowOrchestratorBackend.register(AirflowWorkflowOrchestratorIntegration)
