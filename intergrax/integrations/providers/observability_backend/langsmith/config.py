# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangSmith observability integration configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_LANGSMITH_URL = "INTERGRAX_LANGSMITH_URL"
ENV_LANGSMITH_API_KEY = "INTERGRAX_LANGSMITH_API_KEY"
ENV_LANGSMITH_PROJECT = "INTERGRAX_LANGSMITH_PROJECT"

DEFAULT_TIMEOUT_SECONDS = 30.0


class LangSmithIntegrationConfig(BaseIntegrationConfig):
    base_url: str = "https://api.smith.langchain.com"
    api_key: str = ""
    project: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> LangSmithIntegrationConfig:
        payload: dict[str, object] = {
            "base_url": os.environ.get(ENV_LANGSMITH_URL, "https://api.smith.langchain.com").strip()
            or "https://api.smith.langchain.com",
            "api_key": os.environ.get(ENV_LANGSMITH_API_KEY, "").strip(),
            "project": os.environ.get(ENV_LANGSMITH_PROJECT, "").strip(),
        }
        timeout_raw = os.environ.get("INTERGRAX_LANGSMITH_TIMEOUT_SECONDS", "").strip()
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        payload.update(overrides)
        return cls.model_validate(payload)
