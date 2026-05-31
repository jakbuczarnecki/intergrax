# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa vector store integration configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_VESPA_URL = "INTERGRAX_VESPA_URL"
ENV_VESPA_COLLECTION = "INTERGRAX_VESPA_COLLECTION"
ENV_VESPA_TENANT = "INTERGRAX_VESPA_TENANT"


class VespaIntegrationConfig(BaseIntegrationConfig):
    base_url: str = "http://localhost:8080"
    collection: str = "intergrax"
    tenant_id: str = "default"

    def require_url(self) -> str:
        return self.base_url.rstrip("/")

    @classmethod
    def from_env(cls, **overrides: object) -> VespaIntegrationConfig:
        payload = {
            "base_url": os.environ.get(ENV_VESPA_URL, "http://localhost:8080").strip() or "http://localhost:8080",
            "collection": os.environ.get(ENV_VESPA_COLLECTION, "intergrax").strip() or "intergrax",
            "tenant_id": os.environ.get(ENV_VESPA_TENANT, "default").strip() or "default",
        }
        payload.update(overrides)
        return cls.model_validate(payload)
