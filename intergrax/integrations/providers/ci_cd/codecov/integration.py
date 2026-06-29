# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Codecov ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CODECOV_CI_CD_PROVIDER_ID = "codecov"


class CodecovCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Codecov ci cd integration."""

    pass


CodecovCiCdClient = CiCdBackend

class CodecovCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Codecov ci cd entrypoint.

    Legacy catalog factory (create_codecov_ci_cd) owns catalog behavior; legacy factories use from_client().
    """

    config: CodecovCiCdIntegrationConfig = CodecovCiCdIntegrationConfig()
    _client: CodecovCiCdClient | None = PrivateAttr(default=None)
    

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
            slug=CODECOV_CI_CD_PROVIDER_ID,
            healthy=bool(result),
            detail="codecov ready probe",
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
        client: CodecovCiCdClient,
        *,
        enabled: bool = False,
    ) -> CodecovCiCdIntegration:
        integration = cls.for_provider(
            provider_id=CODECOV_CI_CD_PROVIDER_ID,
            display_name="Codecov",
            config=CodecovCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CodecovCiCdClient | None:
        return self._client

CiCdBackend.register(CodecovCiCdIntegration)
