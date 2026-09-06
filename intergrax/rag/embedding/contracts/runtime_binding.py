# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Typed embedding provider runtime binding contracts (P2-002-B3)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig


class EmbeddingProviderConfigurationError(ValueError):
    """Raised when embedding provider selection inputs conflict or are invalid."""


class EmbeddingProviderRuntimeBindingError(RuntimeError):
    """Raised when catalog runtime binding cannot produce an EmbeddingProvider."""


@dataclass(frozen=True, slots=True)
class EmbeddingProviderRuntimeBindingContext:
    """Inputs for provider-owned runtime construction at bind time."""

    provider_slug: str
    model: str | None = None
    execution_config: EmbeddingProviderExecutionConfig | None = None
    integration_options: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class EmbeddingProviderRuntimeBinder(Protocol):
    """Provider-owned binder that constructs a typed EmbeddingProvider runtime."""

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider: ...


__all__ = [
    "EmbeddingProviderConfigurationError",
    "EmbeddingProviderRuntimeBindingContext",
    "EmbeddingProviderRuntimeBinder",
    "EmbeddingProviderRuntimeBindingError",
]
