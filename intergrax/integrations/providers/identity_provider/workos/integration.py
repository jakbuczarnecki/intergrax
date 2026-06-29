# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Workos identity provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WORKOS_IDENTITY_PROVIDER_PROVIDER_ID = "workos"


class WorkosIdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Workos identity provider integration."""

    pass


WorkosIdentityProviderClient = IdentityProviderBackend

class WorkosIdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Single public Workos identity provider entrypoint.

    Legacy catalog factory (create_workos_identity_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: WorkosIdentityProviderIntegrationConfig = WorkosIdentityProviderIntegrationConfig()
    _client: WorkosIdentityProviderClient | None = PrivateAttr(default=None)
    

    def list_tenants(self, limit: int = 50):
        return self._require_client().list_tenants(limit=limit)

    def userinfo(self, token):
        return self._require_client().userinfo(token)

    def verify_token(self, token):
        return self._require_client().verify_token(token)

    def _require_client(self) -> IdentityProviderBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: WorkosIdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> WorkosIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Workos",
            config=WorkosIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WorkosIdentityProviderClient | None:
        return self._client

IdentityProviderBackend.register(WorkosIdentityProviderIntegration)
