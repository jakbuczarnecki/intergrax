# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Daytona sandbox host integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DAYTONA_SANDBOX_HOST_PROVIDER_ID = "daytona"


class DaytonaSandboxHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Daytona sandbox host integration."""

    pass


DaytonaSandboxHostClient = SandboxHostBackend

class DaytonaSandboxHostIntegration(SandboxHostIntegrationContract):
    """
    Single public Daytona sandbox host entrypoint.

    Legacy catalog factory (create_daytona_sandbox_host) owns catalog behavior; legacy factories use from_client().
    """

    config: DaytonaSandboxHostIntegrationConfig = DaytonaSandboxHostIntegrationConfig()
    _client: DaytonaSandboxHostClient | None = PrivateAttr(default=None)
    

    def create_session(self):
        return self._require_client().create_session()

    def exec(self, session_id, command):
        return self._require_client().exec(session_id, command)

    def upload_artifact(self, session_id, local_path, remote_name):
        return self._require_client().upload_artifact(session_id, local_path, remote_name)

    def _require_client(self) -> SandboxHostBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: DaytonaSandboxHostClient,
        *,
        enabled: bool = False,
    ) -> DaytonaSandboxHostIntegration:
        integration = cls.for_provider(
            provider_id=DAYTONA_SANDBOX_HOST_PROVIDER_ID,
            display_name="Daytona",
            config=DaytonaSandboxHostIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DaytonaSandboxHostClient | None:
        return self._client

SandboxHostBackend.register(DaytonaSandboxHostIntegration)
