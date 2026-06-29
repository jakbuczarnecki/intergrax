# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""E2B sandbox host integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

E2B_SANDBOX_HOST_PROVIDER_ID = "e2b"


class E2bSandboxHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for E2B sandbox host integration."""

    pass


E2bSandboxHostClient = SandboxHostBackend

class E2bSandboxHostIntegration(SandboxHostIntegrationContract):
    """
    Single public E2B sandbox host entrypoint.

    Legacy catalog factory (create_e2b_sandbox_host) owns catalog behavior; legacy factories use from_client().
    """

    config: E2bSandboxHostIntegrationConfig = E2bSandboxHostIntegrationConfig()
    _client: E2bSandboxHostClient | None = PrivateAttr(default=None)
    

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
        client: E2bSandboxHostClient,
        *,
        enabled: bool = False,
    ) -> E2bSandboxHostIntegration:
        integration = cls.for_provider(
            provider_id=E2B_SANDBOX_HOST_PROVIDER_ID,
            display_name="E2B",
            config=E2bSandboxHostIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> E2bSandboxHostClient | None:
        return self._client

SandboxHostBackend.register(E2bSandboxHostIntegration)
