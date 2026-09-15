# © Artur Czarnecki. All rights reserved.

"""Provider resolution contract for delegated execution plugins (P2.1-S2C2)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.contracts.delegated_execution_provider import DelegatedExecutionProvider


@runtime_checkable
class DelegatedExecutionProviderResolver(Protocol):
    """Resolve a provider implementation by stable ``provider_id``."""

    def resolve(
        self,
        provider_id: str,
    ) -> DelegatedExecutionProvider[Any, Any] | None:
        """Return the configured provider or ``None`` when not registered."""
        ...


__all__ = ["DelegatedExecutionProviderResolver"]
