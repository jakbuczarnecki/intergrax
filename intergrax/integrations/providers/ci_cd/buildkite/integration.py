# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Buildkite ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BUILDKITE_CI_CD_PROVIDER_ID = "buildkite"


class BuildkiteCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Buildkite ci cd integration."""

    pass


BuildkiteCiCdClient = CiCdBackend

class BuildkiteCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Buildkite ci cd entrypoint.

    Legacy catalog factory (create_buildkite_ci_cd) owns catalog behavior; legacy factories use from_client().
    """

    config: BuildkiteCiCdIntegrationConfig = BuildkiteCiCdIntegrationConfig()
    _client: BuildkiteCiCdClient | None = PrivateAttr(default=None)
    

    def cancel_workflow_run(self, run_id):
        return self._require_client().cancel_workflow_run(run_id)

    def get_workflow_run(self, run_id):
        return self._require_client().get_workflow_run(run_id)

    def list_check_suites(self, *, ref: str, limit: int = 20):
        return self._require_client().list_check_suites(ref=ref, limit=limit)

    def list_workflow_runs(self, workflow_id: str = '', ref: str = '', limit: int = 20):
        return self._require_client().list_workflow_runs(workflow_id=workflow_id, ref=ref, limit=limit)

    def health(self) -> HealthStatus:
        result = self._require_client().health()
        if isinstance(result, HealthStatus):
            return result
        return HealthStatus(
            slug=BUILDKITE_CI_CD_PROVIDER_ID,
            healthy=bool(result),
            detail="buildkite ready probe",
        )

    def _require_client(self) -> CiCdBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: BuildkiteCiCdClient,
        *,
        enabled: bool = False,
    ) -> BuildkiteCiCdIntegration:
        integration = cls.for_provider(
            provider_id=BUILDKITE_CI_CD_PROVIDER_ID,
            display_name="Buildkite",
            config=BuildkiteCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BuildkiteCiCdClient | None:
        return self._client

CiCdBackend.register(BuildkiteCiCdIntegration)
