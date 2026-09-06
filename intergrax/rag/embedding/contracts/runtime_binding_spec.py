# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Embedding-specific runtime binding descriptor for Integrations catalog contract specs."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.registry.runtime_binding import IntegrationRuntimeBindingSpec
from intergrax.rag.embedding.contracts.runtime_binding import EmbeddingProviderRuntimeBinder


@dataclass(frozen=True, slots=True)
class EmbeddingProviderRuntimeBindingSpec(IntegrationRuntimeBindingSpec):
    """Typed embedding-provider runtime binding metadata."""

    binder: EmbeddingProviderRuntimeBinder


__all__ = ["EmbeddingProviderRuntimeBindingSpec"]
