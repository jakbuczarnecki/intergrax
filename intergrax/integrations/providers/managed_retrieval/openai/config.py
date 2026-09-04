# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenAI managed retrieval provider connection configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIManagedRetrievalConfig:
    """Vendor transport settings for OpenAI hosted file search."""

    api_key: str
    poll_interval_seconds: float = 5.0
    max_poll_attempts: int = 120


def openai_managed_retrieval_config_from_env() -> OpenAIManagedRetrievalConfig | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    poll_raw = os.getenv("INTERGRAX_OPENAI_FILE_POLL_INTERVAL_SECONDS", "5").strip()
    attempts_raw = os.getenv("INTERGRAX_OPENAI_FILE_MAX_POLL_ATTEMPTS", "120").strip()
    try:
        poll_interval = float(poll_raw)
    except ValueError:
        poll_interval = 5.0
    try:
        max_poll_attempts = int(attempts_raw)
    except ValueError:
        max_poll_attempts = 120
    return OpenAIManagedRetrievalConfig(
        api_key=api_key,
        poll_interval_seconds=poll_interval,
        max_poll_attempts=max_poll_attempts,
    )
