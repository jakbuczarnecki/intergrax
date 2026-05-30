# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch observability integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_ELASTICSEARCH_URL = "INTERGRAX_ELASTICSEARCH_URL"
ENV_ELASTICSEARCH_INDEX = "INTERGRAX_ELASTICSEARCH_INDEX"
ENV_ELASTICSEARCH_TIMESTAMP_FIELD = "INTERGRAX_ELASTICSEARCH_TIMESTAMP_FIELD"
ENV_ELASTICSEARCH_USER = "INTERGRAX_ELASTICSEARCH_USER"
ENV_ELASTICSEARCH_PASSWORD = "INTERGRAX_ELASTICSEARCH_PASSWORD"
ENV_ELASTICSEARCH_API_KEY = "INTERGRAX_ELASTICSEARCH_API_KEY"

DEFAULT_BASE_URL = "http://localhost:9200"
DEFAULT_INDEX = "logs-*"
DEFAULT_TIMESTAMP_FIELD = "@timestamp"
DEFAULT_TIMEOUT_SECONDS = 30.0


class ElasticsearchIntegrationConfig(BaseIntegrationConfig):
    """
    Elasticsearch HTTP settings for log/metric search.

    The ``query_instant`` / ``query_range`` ``promql`` argument is interpreted as a
    Lucene ``query_string`` (not PromQL).
    """

    base_url: str = DEFAULT_BASE_URL
    index: str = DEFAULT_INDEX
    timestamp_field: str = DEFAULT_TIMESTAMP_FIELD
    user: str = ""
    password: str = ""
    api_key: str = ""

    @property
    def api_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @classmethod
    def from_env(cls, **overrides: object) -> ElasticsearchIntegrationConfig:
        base_url = os.environ.get(ENV_ELASTICSEARCH_URL, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        index = os.environ.get(ENV_ELASTICSEARCH_INDEX, DEFAULT_INDEX).strip() or DEFAULT_INDEX
        timestamp_field = (
            os.environ.get(ENV_ELASTICSEARCH_TIMESTAMP_FIELD, DEFAULT_TIMESTAMP_FIELD).strip()
            or DEFAULT_TIMESTAMP_FIELD
        )
        user = os.environ.get(ENV_ELASTICSEARCH_USER, "").strip()
        password = os.environ.get(ENV_ELASTICSEARCH_PASSWORD, "").strip()
        api_key = os.environ.get(ENV_ELASTICSEARCH_API_KEY, "").strip()
        timeout_raw = os.environ.get("INTERGRAX_ELASTICSEARCH_TIMEOUT_SECONDS", "").strip()
        payload: dict[str, object] = {
            "base_url": base_url,
            "index": index,
            "timestamp_field": timestamp_field,
            "user": user,
            "password": password,
            "api_key": api_key,
        }
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        else:
            payload["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
