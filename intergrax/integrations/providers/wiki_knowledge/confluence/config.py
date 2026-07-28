# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki integration configuration (Phase M.6)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_CONFLUENCE_BASE_URL = "INTERGRAX_CONFLUENCE_BASE_URL"
ENV_CONFLUENCE_EMAIL = "INTERGRAX_CONFLUENCE_EMAIL"
ENV_CONFLUENCE_API_TOKEN = "INTERGRAX_CONFLUENCE_API_TOKEN"

DEFAULT_TIMEOUT_SECONDS = 30.0


class ConfluenceIntegrationConfig(BaseIntegrationConfig):
    """
    Confluence Cloud REST settings.

    ``base_url`` example: ``https://your-domain.atlassian.net/wiki`` (no trailing slash).
    """

    base_url: str = ""
    email: str = ""
    api_token: str = ""

    @property
    def api_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/rest/api"

    @property
    def v2_api_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/v2"

    def v2_api_url(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("path must start with /")
        if "://" in path or "@" in path:
            raise ValueError("path must not contain scheme, host or credentials")
        return f"{self.v2_api_base_url}{path}"

    def page_url(self, page_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"

    @classmethod
    def from_env(cls, **overrides: object) -> ConfluenceIntegrationConfig:
        base_url = os.environ.get(ENV_CONFLUENCE_BASE_URL, "").strip()
        email = os.environ.get(ENV_CONFLUENCE_EMAIL, "").strip()
        api_token = os.environ.get(ENV_CONFLUENCE_API_TOKEN, "").strip()
        timeout_raw = os.environ.get("INTERGRAX_CONFLUENCE_TIMEOUT_SECONDS", "").strip()
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
