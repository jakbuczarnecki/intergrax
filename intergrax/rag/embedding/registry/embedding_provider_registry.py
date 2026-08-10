# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Dict

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class EmbeddingProviderDependencyError(RuntimeError):
    """Raised when a selected provider cannot load its optional dependency."""


class _LazyProviderFactory:
    def __init__(self, factory: Callable[[], EmbeddingProvider]) -> None:
        self.factory = factory


class EmbeddingProviderRegistry:

    def __init__(self, providers: Iterable[EmbeddingProvider] | None = None) -> None:
        self._providers: Dict[
            str, EmbeddingProvider | _LazyProviderFactory
        ] = {}

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

    def register_factory(
        self,
        name: str,
        factory: Callable[[], EmbeddingProvider],
    ) -> None:
        if name in self._providers:
            raise ValueError(
                f"Embedding provider already registered: {name}"
            )

        self._providers[name] = _LazyProviderFactory(factory)

    def get(self, name: str) -> EmbeddingProvider:

        registration = self._providers.get(name)

        if registration is None:
            raise RuntimeError(
                f"Embedding provider not registered: {name}"
            )

        if isinstance(registration, _LazyProviderFactory):
            provider = registration.factory()
            if provider.provider_name() != name:
                raise RuntimeError(
                    f"Embedding provider factory returned the wrong provider for: {name}"
                )
            self._providers[name] = provider
            return provider

        provider = registration
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


def lazy_import_provider_factory(
    *,
    provider_id: str,
    module_name: str,
    class_name: str,
    dependency_name: str,
) -> Callable[[], EmbeddingProvider]:
    """Build a provider factory whose implementation is imported on selection."""

    def create_provider() -> EmbeddingProvider:
        try:
            module = import_module(module_name)
            provider_type = getattr(module, class_name)
        except (ImportError, AttributeError):
            raise EmbeddingProviderDependencyError(
                f"Embedding provider '{provider_id}' requires dependency "
                f"'{dependency_name}'. Install it before selecting this provider."
            ) from None

        return provider_type()

    return create_provider