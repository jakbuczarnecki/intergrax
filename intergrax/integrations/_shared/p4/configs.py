# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.8 integration config models."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class OpenSearchIntegrationConfig(BaseIntegrationConfig):
    """OpenSearch HTTP settings (Elasticsearch-compatible API)."""

    base_url: str = "http://localhost:9200"
    index: str = "logs-*"
    timestamp_field: str = "@timestamp"
    user: str = ""
    password: str = ""
    api_key: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> OpenSearchIntegrationConfig:
        payload = {
            "base_url": _env("INTERGRAX_OPENSEARCH_URL", "http://localhost:9200") or "http://localhost:9200",
            "index": _env("INTERGRAX_OPENSEARCH_INDEX", "logs-*") or "logs-*",
            "timestamp_field": _env("INTERGRAX_OPENSEARCH_TIMESTAMP_FIELD", "@timestamp") or "@timestamp",
            "user": _env("INTERGRAX_OPENSEARCH_USER"),
            "password": _env("INTERGRAX_OPENSEARCH_PASSWORD"),
            "api_key": _env("INTERGRAX_OPENSEARCH_API_KEY"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
