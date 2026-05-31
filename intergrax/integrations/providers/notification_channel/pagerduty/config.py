# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PagerDuty notification channel configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_PAGERDUTY_URL = "INTERGRAX_PAGERDUTY_URL"
ENV_PAGERDUTY_ROUTING_KEY = "INTERGRAX_PAGERDUTY_ROUTING_KEY"


class PagerDutyIntegrationConfig(BaseIntegrationConfig):
    base_url: str = "https://events.pagerduty.com"
    routing_key: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> PagerDutyIntegrationConfig:
        payload = {
            "base_url": os.environ.get(ENV_PAGERDUTY_URL, "https://events.pagerduty.com").strip()
            or "https://events.pagerduty.com",
            "routing_key": (
                os.environ.get(ENV_PAGERDUTY_ROUTING_KEY, "").strip()
                or os.environ.get("INTERGRAX_PAGERDUTY_API_KEY", "").strip()
                or os.environ.get("INTERGRAX_PAGERDUTY_TOKEN", "").strip()
            ),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
