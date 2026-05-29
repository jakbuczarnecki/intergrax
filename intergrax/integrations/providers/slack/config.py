# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_SLACK_WEBHOOK_URL = "INTERGRAX_SLACK_WEBHOOK_URL"
ENV_SLACK_SIGNING_SECRET = "INTERGRAX_SLACK_SIGNING_SECRET"

DEFAULT_WEBHOOK_URL = ""


class SlackIntegrationConfig(BaseIntegrationConfig):
    webhook_url: str = DEFAULT_WEBHOOK_URL
    signing_secret: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> SlackIntegrationConfig:
        webhook_url = os.environ.get(ENV_SLACK_WEBHOOK_URL, DEFAULT_WEBHOOK_URL).strip()
        signing_secret = os.environ.get(ENV_SLACK_SIGNING_SECRET, "").strip()
        payload: dict[str, object] = {
            "webhook_url": webhook_url,
            "signing_secret": signing_secret,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
