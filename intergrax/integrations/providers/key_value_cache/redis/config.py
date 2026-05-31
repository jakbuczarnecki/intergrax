# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Redis integration configuration (Phase M.4)."""

from __future__ import annotations

import os
from typing import Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_REDIS_URL = "INTERGRAX_REDIS_URL"
ENV_REDIS_DB = "INTERGRAX_REDIS_DB"
ENV_REDIS_KEY_PREFIX = "INTERGRAX_REDIS_KEY_PREFIX"

DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class RedisIntegrationConfig(BaseIntegrationConfig):
    url: str = DEFAULT_REDIS_URL
    db: int = 0
    decode_responses: bool = False

    @classmethod
    def from_env(cls, **overrides: object) -> RedisIntegrationConfig:
        url = os.environ.get(ENV_REDIS_URL, DEFAULT_REDIS_URL).strip() or DEFAULT_REDIS_URL
        db_raw = os.environ.get(ENV_REDIS_DB, "").strip()
        prefix = os.environ.get(ENV_REDIS_KEY_PREFIX, "intergrax").strip() or "intergrax"
        db = int(db_raw) if db_raw else 0
        payload = {
            "url": url,
            "db": db,
            "key_prefix": prefix,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
