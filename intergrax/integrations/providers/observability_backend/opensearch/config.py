# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenSearch observability integration configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig


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
            "base_url": os.environ.get("INTERGRAX_OPENSEARCH_URL", "http://localhost:9200").strip()
            or "http://localhost:9200",
            "index": os.environ.get("INTERGRAX_OPENSEARCH_INDEX", "logs-*").strip() or "logs-*",
            "timestamp_field": os.environ.get("INTERGRAX_OPENSEARCH_TIMESTAMP_FIELD", "@timestamp").strip()
            or "@timestamp",
            "user": os.environ.get("INTERGRAX_OPENSEARCH_USER", "").strip(),
            "password": os.environ.get("INTERGRAX_OPENSEARCH_PASSWORD", "").strip(),
            "api_key": os.environ.get("INTERGRAX_OPENSEARCH_API_KEY", "").strip(),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
