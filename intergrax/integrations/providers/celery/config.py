# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Celery integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_CELERY_BROKER_URL = "INTERGRAX_CELERY_BROKER_URL"
ENV_CELERY_BACKEND_URL = "INTERGRAX_CELERY_BACKEND_URL"
ENV_CELERY_APP_NAME = "INTERGRAX_CELERY_APP_NAME"

DEFAULT_BROKER_URL = "redis://localhost:6379/1"
DEFAULT_BACKEND_URL = "redis://localhost:6379/2"
DEFAULT_APP_NAME = "intergrax"


class CeleryIntegrationConfig(BaseIntegrationConfig):
    app_name: str = DEFAULT_APP_NAME
    broker_url: str = DEFAULT_BROKER_URL
    backend_url: str = DEFAULT_BACKEND_URL

    @classmethod
    def from_env(cls, **overrides: object) -> CeleryIntegrationConfig:
        broker = (
            os.environ.get(ENV_CELERY_BROKER_URL, DEFAULT_BROKER_URL).strip()
            or DEFAULT_BROKER_URL
        )
        backend = (
            os.environ.get(ENV_CELERY_BACKEND_URL, DEFAULT_BACKEND_URL).strip()
            or DEFAULT_BACKEND_URL
        )
        app_name = (
            os.environ.get(ENV_CELERY_APP_NAME, DEFAULT_APP_NAME).strip()
            or DEFAULT_APP_NAME
        )
        payload: dict[str, object] = {
            "broker_url": broker,
            "backend_url": backend,
            "app_name": app_name,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
