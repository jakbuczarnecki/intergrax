# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sentry observability integration configuration (OBS-SENTRY-1)."""

from __future__ import annotations

import os
from typing import Self

from pydantic import model_validator

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_SENTRY_DSN = "INTERGRAX_SENTRY_DSN"
ENV_SENTRY_ENVIRONMENT = "INTERGRAX_SENTRY_ENVIRONMENT"
ENV_SENTRY_RELEASE = "INTERGRAX_SENTRY_RELEASE"
ENV_SENTRY_SERVER_NAME = "INTERGRAX_SENTRY_SERVER_NAME"
ENV_SENTRY_SHUTDOWN_TIMEOUT_SECONDS = "INTERGRAX_SENTRY_SHUTDOWN_TIMEOUT_SECONDS"
ENV_SENTRY_DEBUG = "INTERGRAX_SENTRY_DEBUG"

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 2.0


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class SentryIntegrationConfig(BaseIntegrationConfig):
    """Provider-owned Sentry SDK settings — DSN and secrets stay out of vendor payloads."""

    dsn: str = ""
    environment: str = ""
    release: str = ""
    server_name: str = ""
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    send_default_pii: bool = False
    attach_stacktrace: bool = False
    debug: bool = False

    @model_validator(mode="after")
    def _validate_shutdown_timeout(self) -> Self:
        if self.shutdown_timeout_seconds < 0:
            msg = "shutdown_timeout_seconds must be >= 0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_env(cls, **overrides: object) -> SentryIntegrationConfig:
        payload: dict[str, object] = {
            "dsn": os.environ.get(ENV_SENTRY_DSN, "").strip(),
            "environment": os.environ.get(ENV_SENTRY_ENVIRONMENT, "").strip(),
            "release": os.environ.get(ENV_SENTRY_RELEASE, "").strip(),
            "server_name": os.environ.get(ENV_SENTRY_SERVER_NAME, "").strip(),
            "debug": _parse_bool(os.environ.get(ENV_SENTRY_DEBUG, "")),
        }
        timeout_raw = os.environ.get(ENV_SENTRY_SHUTDOWN_TIMEOUT_SECONDS, "").strip()
        if timeout_raw:
            payload["shutdown_timeout_seconds"] = float(timeout_raw)
        else:
            payload["shutdown_timeout_seconds"] = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
