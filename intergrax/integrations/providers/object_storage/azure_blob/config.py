# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Blob object storage configuration (Phase M.6 P2)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

ENV_AZURE_BLOB_CONTAINER = "INTERGRAX_AZURE_BLOB_CONTAINER"
ENV_AZURE_BLOB_PREFIX = "INTERGRAX_AZURE_BLOB_PREFIX"
ENV_AZURE_BLOB_CONNECTION_STRING = "INTERGRAX_AZURE_BLOB_CONNECTION_STRING"
ENV_AZURE_BLOB_ACCOUNT_URL = "INTERGRAX_AZURE_BLOB_ACCOUNT_URL"


class AzureBlobIntegrationConfig(BaseIntegrationConfig):
    container: str = ""
    prefix: str = ""
    connection_string: str = ""
    account_url: str = ""

    def require_container(self) -> str:
        name = (self.container or "").strip()
        if not name:
            raise IntegrationConfigurationError(
                "Azure Blob requires container (INTERGRAX_AZURE_BLOB_CONTAINER)"
            )
        return name

    def object_key(self, key: str) -> str:
        normalized = key.lstrip("/")
        prefix = self.prefix.strip("/")
        if prefix:
            return f"{prefix}/{normalized}"
        return normalized

    @classmethod
    def from_env(cls, **overrides: object) -> AzureBlobIntegrationConfig:
        payload: dict[str, object] = {
            "container": os.environ.get(ENV_AZURE_BLOB_CONTAINER, "").strip(),
            "prefix": os.environ.get(ENV_AZURE_BLOB_PREFIX, "").strip(),
            "connection_string": os.environ.get(ENV_AZURE_BLOB_CONNECTION_STRING, "").strip(),
            "account_url": os.environ.get(ENV_AZURE_BLOB_ACCOUNT_URL, "").strip(),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
