# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Teams integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_TEAMS_WEBHOOK_URL = "INTERGRAX_TEAMS_WEBHOOK_URL"
ENV_TEAMS_SECURITY_TOKEN = "INTERGRAX_TEAMS_SECURITY_TOKEN"

DEFAULT_WEBHOOK_URL = ""


class TeamsIntegrationConfig(BaseIntegrationConfig):
    webhook_url: str = DEFAULT_WEBHOOK_URL
    security_token: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> TeamsIntegrationConfig:
        webhook_url = os.environ.get(ENV_TEAMS_WEBHOOK_URL, DEFAULT_WEBHOOK_URL).strip()
        security_token = os.environ.get(ENV_TEAMS_SECURITY_TOKEN, "").strip()
        payload: dict[str, object] = {
            "webhook_url": webhook_url,
            "security_token": security_token,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
