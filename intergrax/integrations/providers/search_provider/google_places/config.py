# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig


class GooglePlacesIntegrationConfig(BaseIntegrationConfig):
    api_key: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> GooglePlacesIntegrationConfig:
        payload: dict[str, object] = {
            "api_key": os.environ.get("GOOGLE_PLACES_API_KEY", "").strip(),
        }
        timeout_raw = os.environ.get("INTERGRAX_GOOGLE_PLACES_TIMEOUT_SECONDS", "").strip()
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        payload.update(overrides)
        return cls.model_validate(payload)
