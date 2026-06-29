# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Github Actions ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GITHUB_ACTIONS_CI_CD_PROVIDER_ID = "github_actions"


class GithubActionsCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Github Actions ci cd integration."""

    pass


GithubActionsCiCdClient = CiCdBackend

class GithubActionsCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Github Actions ci cd entrypoint.

    Legacy catalog factory (create_github_actions_ci_cd) owns catalog behavior; legacy factories use from_client().
    """

    config: GithubActionsCiCdIntegrationConfig = GithubActionsCiCdIntegrationConfig()
    _client: GithubActionsCiCdClient | None = PrivateAttr(default=None)
    

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
        client: GithubActionsCiCdClient,
        *,
        enabled: bool = False,
    ) -> GithubActionsCiCdIntegration:
        integration = cls.for_provider(
            provider_id=GITHUB_ACTIONS_CI_CD_PROVIDER_ID,
            display_name="Github Actions",
            config=GithubActionsCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GithubActionsCiCdClient | None:
        return self._client

CiCdBackend.register(GithubActionsCiCdIntegration)
