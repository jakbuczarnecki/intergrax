# © Artur Czarnecki. All rights reserved.

"""Configuration helpers for OpenAI vector store tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpenAIVectorStoreToolConfig:
    """Resolved defaults for OpenAI hosted vector store operations."""

    vector_store_id: Optional[str]
    default_model: str = "gpt-4o-mini"
    poll_interval_seconds: float = 5.0
    max_poll_attempts: int = 120


def openai_vector_store_config_from_env() -> OpenAIVectorStoreToolConfig:
    return OpenAIVectorStoreToolConfig(
        vector_store_id=os.getenv("INTERGRAX_OPENAI_VECTOR_STORE_ID", "").strip() or None,
        default_model=os.getenv("INTERGRAX_OPENAI_FILE_SEARCH_MODEL", "gpt-4o-mini").strip()
        or "gpt-4o-mini",
    )
