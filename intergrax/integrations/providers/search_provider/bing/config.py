# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bing Web Search integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_BING_API_KEY = "INTERGRAX_BING_API_KEY"

LEGACY_ENV_API_KEY = "BING_SEARCH_V7_API_KEY"

DEFAULT_TIMEOUT_SECONDS = 20.0


class BingIntegrationConfig(BaseIntegrationConfig):
    api_key: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> BingIntegrationConfig:
        api_key = (
            os.environ.get(ENV_BING_API_KEY)
            or os.environ.get(LEGACY_ENV_API_KEY)
            or ""
        ).strip()
        timeout_raw = os.environ.get("INTERGRAX_BING_TIMEOUT_SECONDS", "").strip()
        payload: dict[str, object] = {"api_key": api_key}
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        else:
            payload["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
