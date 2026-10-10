# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa vector store integration configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

ENV_VESPA_URL = "INTERGRAX_VESPA_URL"
ENV_VESPA_COLLECTION = "INTERGRAX_VESPA_COLLECTION"
ENV_VESPA_TENANT = "INTERGRAX_VESPA_TENANT"


class VespaIntegrationConfig(BaseIntegrationConfig):
    base_url: str = ""
    collection: str = "intergrax"
    tenant_id: str = ""

    def require_url(self) -> str:
        url = self.base_url.strip()
        if not url:
            raise IntegrationConfigurationError(
                "Vespa integration requires explicit base_url configuration (INTERGRAX_VESPA_URL)",
            )
        return url.rstrip("/")

    def require_tenant_id(self) -> str:
        tenant = self.tenant_id.strip()
        if not tenant:
            raise IntegrationConfigurationError(
                "Vespa integration requires explicit tenant_id configuration (INTERGRAX_VESPA_TENANT)",
            )
        return tenant

    @classmethod
    def from_env(cls, **overrides: object) -> VespaIntegrationConfig:
        tenant_override = overrides.get("tenant_id")
        tenant_from_env = os.environ.get(ENV_VESPA_TENANT, "").strip()
        if tenant_override is not None:
            tenant_id = str(tenant_override).strip()
        elif tenant_from_env:
            tenant_id = tenant_from_env
        else:
            raise IntegrationConfigurationError(
                f"Vespa integration requires {ENV_VESPA_TENANT}",
            )

        url_override = overrides.get("base_url")
        url_from_env = os.environ.get(ENV_VESPA_URL, "").strip()
        if url_override is not None:
            base_url = str(url_override).strip()
        elif url_from_env:
            base_url = url_from_env
        else:
            raise IntegrationConfigurationError(
                f"Vespa integration requires {ENV_VESPA_URL}",
            )

        payload = {
            "base_url": base_url,
            "collection": os.environ.get(ENV_VESPA_COLLECTION, "intergrax").strip() or "intergrax",
            "tenant_id": tenant_id,
        }
        payload.update(overrides)
        config = cls.model_validate(payload)
        config.require_url()
        config.require_tenant_id()
        return config
