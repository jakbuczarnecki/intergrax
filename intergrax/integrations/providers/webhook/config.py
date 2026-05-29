# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic HTTP webhook notification integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_WEBHOOK_URL = "INTERGRAX_WEBHOOK_URL"

DEFAULT_WEBHOOK_URL = ""


class WebhookIntegrationConfig(BaseIntegrationConfig):
    webhook_url: str = DEFAULT_WEBHOOK_URL

    @classmethod
    def from_env(cls, **overrides: object) -> WebhookIntegrationConfig:
        webhook_url = os.environ.get(ENV_WEBHOOK_URL, DEFAULT_WEBHOOK_URL).strip()
        payload: dict[str, object] = {"webhook_url": webhook_url}
        payload.update(overrides)
        return cls.model_validate(payload)
