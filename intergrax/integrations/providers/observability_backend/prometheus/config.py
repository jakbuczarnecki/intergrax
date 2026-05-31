# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prometheus observability integration configuration (Phase M.6)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_PROMETHEUS_BASE_URL = "INTERGRAX_PROMETHEUS_BASE_URL"
ENV_PROMETHEUS_BEARER_TOKEN = "INTERGRAX_PROMETHEUS_BEARER_TOKEN"

DEFAULT_BASE_URL = "http://localhost:9090"
DEFAULT_TIMEOUT_SECONDS = 30.0


class PrometheusIntegrationConfig(BaseIntegrationConfig):
    """
    Prometheus HTTP API settings.

    ``base_url`` example: ``http://prometheus:9090`` (no trailing slash).
    """

    base_url: str = DEFAULT_BASE_URL
    bearer_token: str = ""

    @property
    def api_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @classmethod
    def from_env(cls, **overrides: object) -> PrometheusIntegrationConfig:
        base_url = os.environ.get(ENV_PROMETHEUS_BASE_URL, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        bearer_token = os.environ.get(ENV_PROMETHEUS_BEARER_TOKEN, "").strip()
        timeout_raw = os.environ.get("INTERGRAX_PROMETHEUS_TIMEOUT_SECONDS", "").strip()
        payload: dict[str, object] = {
            "base_url": base_url,
            "bearer_token": bearer_token,
        }
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        else:
            payload["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
