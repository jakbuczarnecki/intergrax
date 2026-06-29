# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Circleci ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CIRCLECI_CI_CD_PROVIDER_ID = "circleci"


class CircleciCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Circleci ci cd integration."""

    pass


CircleciCiCdClient = CiCdBackend

class CircleciCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Circleci ci cd entrypoint.

    Legacy catalog factory (create_circleci_ci_cd) owns catalog behavior; legacy factories use from_client().
    """

    config: CircleciCiCdIntegrationConfig = CircleciCiCdIntegrationConfig()
    _client: CircleciCiCdClient | None = PrivateAttr(default=None)
    

    def cancel_workflow_run(self, run_id):
        return self._require_client().cancel_workflow_run(run_id)

    def get_workflow_run(self, run_id):
        return self._require_client().get_workflow_run(run_id)

    def list_check_suites(self, ref, limit: int = 20):
        return self._require_client().list_check_suites(ref, limit=limit)

    def list_workflow_runs(self, workflow_id: str = '', ref: str = '', limit: int = 20):
        return self._require_client().list_workflow_runs(workflow_id=workflow_id, ref=ref, limit=limit)

    def _require_client(self) -> CiCdBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: CircleciCiCdClient,
        *,
        enabled: bool = False,
    ) -> CircleciCiCdIntegration:
        integration = cls.for_provider(
            provider_id=CIRCLECI_CI_CD_PROVIDER_ID,
            display_name="Circleci",
            config=CircleciCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CircleciCiCdClient | None:
        return self._client

CiCdBackend.register(CircleciCiCdIntegration)
