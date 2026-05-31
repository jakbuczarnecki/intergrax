# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig


class RedditIntegrationConfig(BaseIntegrationConfig):
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "IntegraxWebSearch/1.0 (by intergrax.ai)"
    include_comments: bool = True
    comments_limit: int = 10

    @classmethod
    def from_env(cls, **overrides: object) -> RedditIntegrationConfig:
        payload: dict[str, object] = {
            "client_id": os.environ.get("REDDIT_CLIENT_ID", "").strip(),
            "client_secret": os.environ.get("REDDIT_CLIENT_SECRET", "").strip(),
            "user_agent": os.environ.get("REDDIT_USER_AGENT", "IntegraxWebSearch/1.0 (by intergrax.ai)").strip(),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
