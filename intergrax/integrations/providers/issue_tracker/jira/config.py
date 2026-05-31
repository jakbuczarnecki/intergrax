# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira issue tracker integration configuration (Phase M.6)."""

from __future__ import annotations

import os
from typing import Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_JIRA_BASE_URL = "INTERGRAX_JIRA_BASE_URL"
ENV_JIRA_EMAIL = "INTERGRAX_JIRA_EMAIL"
ENV_JIRA_API_TOKEN = "INTERGRAX_JIRA_API_TOKEN"

DEFAULT_TIMEOUT_SECONDS = 30.0


class JiraIntegrationConfig(BaseIntegrationConfig):
    """
    Jira Cloud / Server REST settings.

    ``base_url`` example: ``https://your-domain.atlassian.net`` (no trailing slash).
    """

    base_url: str = ""
    email: str = ""
    api_token: str = ""

    @property
    def api_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/rest/api/3"

    def issue_url(self, issue_key: str) -> str:
        return f"{self.base_url.rstrip('/')}/browse/{issue_key}"

    @classmethod
    def from_env(cls, **overrides: object) -> JiraIntegrationConfig:
        base_url = os.environ.get(ENV_JIRA_BASE_URL, "").strip()
        email = os.environ.get(ENV_JIRA_EMAIL, "").strip()
        api_token = os.environ.get(ENV_JIRA_API_TOKEN, "").strip()
        timeout_raw = os.environ.get("INTERGRAX_JIRA_TIMEOUT_SECONDS", "").strip()
        payload: dict[str, object] = {
            "base_url": base_url,
            "email": email,
            "api_token": api_token,
        }
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        else:
            payload["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
