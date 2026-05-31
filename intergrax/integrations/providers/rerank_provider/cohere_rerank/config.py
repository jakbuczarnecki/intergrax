# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class CohereRerankIntegrationConfig(BaseModel):
    model: str = Field(
        default_factory=lambda: os.getenv(
            "INTERGRAX_DEFAULT_COHERE_RERANK_MODEL",
            "rerank-english-v3.0",
        )
    )
    api_key: str | None = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    top_n: int | None = None

    @classmethod
    def from_env(cls, **overrides: object) -> CohereRerankIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
