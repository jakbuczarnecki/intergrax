# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory helpers for OpenAI managed retrieval adapter."""

from __future__ import annotations

from intergrax.integrations.contracts.managed_retrieval import ManagedRetrievalBackend
from intergrax.integrations.providers.managed_retrieval.openai.adapter import (
    create_openai_managed_retrieval_adapter,
)
from intergrax.integrations.providers.managed_retrieval.openai.config import (
    OpenAIManagedRetrievalConfig,
    openai_managed_retrieval_config_from_env,
)

__all__ = [
    "create_openai_managed_retrieval",
    "try_create_openai_managed_retrieval_from_env",
]


def create_openai_managed_retrieval(
    config: OpenAIManagedRetrievalConfig,
) -> ManagedRetrievalBackend:
    return create_openai_managed_retrieval_adapter(config)


def try_create_openai_managed_retrieval_from_env() -> ManagedRetrievalBackend | None:
    config = openai_managed_retrieval_config_from_env()
    if config is None:
        return None
    return create_openai_managed_retrieval(config)
