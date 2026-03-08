# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class EmbeddingProviderRegistry:

    def __init__(self, providers: Iterable[EmbeddingProvider] | None = None):
        self._providers: Dict[str, EmbeddingProvider] = {}

        if providers:
            for provider in providers:
                self.register(provider)

    def register(self, provider: EmbeddingProvider) -> None:

        name = provider.provider_name()

        if name in self._providers:
            raise ValueError(
                f"Embedding provider already registered: {name}"
            )

        self._providers[name] = provider

    def get(self, name: str) -> EmbeddingProvider:

        provider = self._providers.get(name)

        if provider is None:
            raise RuntimeError(
                f"Embedding provider not registered: {name}"
            )

        return provider
    

    def default_provider(self) -> str:
        """
        Returns identifier of the default provider.

        The default provider is defined as the first provider
        registered in the registry. This ensures deterministic
        bootstrap behaviour.
        """

        if not self._providers:
            raise RuntimeError("No embedding providers registered.")

        return next(iter(self._providers.keys()))