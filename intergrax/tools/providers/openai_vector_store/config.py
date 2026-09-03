# © Artur Czarnecki. All rights reserved.

"""Tool-level defaults for OpenAI managed vector store catalog tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpenAIVectorStoreToolConfig:
    """Application/tool semantics for OpenAI hosted vector store tools."""

    vector_store_id: Optional[str]
    default_model: str = "gpt-4o-mini"


def openai_vector_store_config_from_env() -> OpenAIVectorStoreToolConfig:
    return OpenAIVectorStoreToolConfig(
        vector_store_id=os.getenv("INTERGRAX_OPENAI_VECTOR_STORE_ID", "").strip() or None,
        default_model=os.getenv("INTERGRAX_OPENAI_FILE_SEARCH_MODEL", "gpt-4o-mini").strip()
        or "gpt-4o-mini",
    )
