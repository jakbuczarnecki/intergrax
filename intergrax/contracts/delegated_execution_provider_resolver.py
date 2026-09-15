# © Artur Czarnecki. All rights reserved.

"""Provider resolution contract for delegated execution plugins (P2.1-S2C2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.delegated_execution_provider import DelegatedExecutionCapabilities


@runtime_checkable
class DelegatedExecutionProviderHandle(Protocol):
    """Minimal provider identity surface for status/control resolution."""

    @property
    def provider_id(self) -> str:
        """Stable provider identifier."""
        ...

    @property
    def provider_version(self) -> str:
        """Provider implementation version."""
        ...

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        """Declared provider capabilities."""
        ...


@runtime_checkable
class DelegatedExecutionProviderResolver(Protocol):
    """Resolve a provider implementation by stable ``provider_id``."""

    def resolve(
        self,
        provider_id: str,
    ) -> DelegatedExecutionProviderHandle | None:
        """Return the configured provider or ``None`` when not registered."""
        ...


__all__ = [
    "DelegatedExecutionProviderHandle",
    "DelegatedExecutionProviderResolver",
]
