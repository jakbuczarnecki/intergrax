# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Custom Search (CSE) integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_GOOGLE_CSE_API_KEY = "INTERGRAX_GOOGLE_CSE_API_KEY"
ENV_GOOGLE_CSE_CX = "INTERGRAX_GOOGLE_CSE_CX"

LEGACY_ENV_API_KEY = "GOOGLE_CSE_API_KEY"
LEGACY_ENV_CX = "GOOGLE_CSE_CX"

DEFAULT_TIMEOUT_SECONDS = 20.0


class GoogleCSEIntegrationConfig(BaseIntegrationConfig):
    api_key: str = ""
    cx: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> GoogleCSEIntegrationConfig:
        api_key = (
            os.environ.get(ENV_GOOGLE_CSE_API_KEY)
            or os.environ.get(LEGACY_ENV_API_KEY)
            or ""
        ).strip()
        cx = (
            os.environ.get(ENV_GOOGLE_CSE_CX)
            or os.environ.get(LEGACY_ENV_CX)
            or ""
        ).strip()
        timeout_raw = os.environ.get("INTERGRAX_GOOGLE_CSE_TIMEOUT_SECONDS", "").strip()
        payload: dict[str, object] = {
            "api_key": api_key,
            "cx": cx,
        }
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        else:
            payload["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
