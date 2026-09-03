# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition helpers for managed retrieval provider materialization."""

from __future__ import annotations

from intergrax.integrations.contracts.managed_retrieval import ManagedRetrievalBackend


def try_create_managed_retrieval_from_env() -> ManagedRetrievalBackend | None:
    """Materialize the default shipped managed retrieval adapter when env credentials exist."""
    from intergrax.integrations.providers.managed_retrieval.openai.bundle import (
        try_create_openai_managed_retrieval_from_env,
    )

    return try_create_openai_managed_retrieval_from_env()
