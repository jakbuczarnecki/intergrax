# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class JinaRerankIntegrationConfig(BaseModel):
    model: str = Field(
        default_factory=lambda: os.getenv(
            "INTERGRAX_DEFAULT_JINA_RERANK_MODEL",
            "jina-reranker-v1-base-en",
        )
    )
    api_key: str | None = Field(default_factory=lambda: os.getenv("JINA_API_KEY"))
    api_url: str = "https://api.jina.ai/v1/rerank"

    @classmethod
    def from_env(cls, **overrides: object) -> JinaRerankIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
