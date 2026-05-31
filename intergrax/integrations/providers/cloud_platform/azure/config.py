# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure cloud platform integration configuration (Phase M.6)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_AZURE_TENANT_ID = "INTERGRAX_AZURE_TENANT_ID"
ENV_AZURE_CLIENT_ID = "INTERGRAX_AZURE_CLIENT_ID"
ENV_AZURE_CLIENT_SECRET = "INTERGRAX_AZURE_CLIENT_SECRET"
ENV_AZURE_SUBSCRIPTION_ID = "INTERGRAX_AZURE_SUBSCRIPTION_ID"
ENV_AZURE_LOCATION = "INTERGRAX_AZURE_LOCATION"

AZURE_MANAGEMENT_SCOPE = "https://management.azure.com/.default"


class AzureIntegrationConfig(BaseIntegrationConfig):
    """
    Azure auth settings for the cloud platform facade.

    When ``tenant_id``, ``client_id``, and ``client_secret`` are set, a service
    principal credential is used; otherwise ``DefaultAzureCredential`` (MI, CLI, env).
    """

    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    subscription_id: str = ""
    location: str = ""

    @property
    def uses_service_principal(self) -> bool:
        return bool(self.tenant_id and self.client_id and self.client_secret)

    @classmethod
    def from_env(cls, **overrides: object) -> AzureIntegrationConfig:
        payload: dict[str, object] = {
            "tenant_id": os.environ.get(ENV_AZURE_TENANT_ID, "").strip(),
            "client_id": os.environ.get(ENV_AZURE_CLIENT_ID, "").strip(),
            "client_secret": os.environ.get(ENV_AZURE_CLIENT_SECRET, "").strip(),
            "subscription_id": os.environ.get(ENV_AZURE_SUBSCRIPTION_ID, "").strip(),
            "location": os.environ.get(ENV_AZURE_LOCATION, "").strip(),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
