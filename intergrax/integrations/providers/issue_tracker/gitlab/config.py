# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GitLab issue tracker integration configuration (Phase M.8 full adapter)."""

from __future__ import annotations

import os
from urllib.parse import quote

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_GITLAB_URL = "INTERGRAX_GITLAB_URL"
ENV_GITLAB_TOKEN = "INTERGRAX_GITLAB_TOKEN"
ENV_GITLAB_REPO = "INTERGRAX_GITLAB_REPO"

DEFAULT_TIMEOUT_SECONDS = 30.0


class GitLabIntegrationConfig(BaseIntegrationConfig):
    """GitLab REST API v4 settings."""

    base_url: str = "https://gitlab.com"
    token: str = ""
    project_id: str = ""

    @property
    def api_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/v4"

    def encoded_project(self) -> str:
        return quote(self.project_id, safe="")

    def issue_url(self, project_id: str, iid: str) -> str:
        return f"{self.base_url.rstrip('/')}/{project_id}/-/issues/{iid}"

    @classmethod
    def from_env(cls, **overrides: object) -> GitLabIntegrationConfig:
        payload: dict[str, object] = {
            "base_url": os.environ.get(ENV_GITLAB_URL, "https://gitlab.com").strip() or "https://gitlab.com",
            "token": os.environ.get(ENV_GITLAB_TOKEN, "").strip(),
            "project_id": (
                os.environ.get(ENV_GITLAB_REPO, "").strip()
                or os.environ.get("INTERGRAX_GITLAB_ORG", "").strip()
            ),
        }
        timeout_raw = os.environ.get("INTERGRAX_GITLAB_TIMEOUT_SECONDS", "").strip()
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        payload.update(overrides)
        return cls.model_validate(payload)
