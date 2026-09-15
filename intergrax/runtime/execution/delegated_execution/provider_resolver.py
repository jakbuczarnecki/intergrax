# © Artur Czarnecki. All rights reserved.

"""Injectable delegated execution provider resolver (P2.1-S2C2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.contracts.delegated_execution_provider import DelegatedExecutionProvider
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)


class MappingDelegatedExecutionProviderResolver(DelegatedExecutionProviderResolver):
    """Composition-time provider map — not a global service locator."""

    __slots__ = ("_providers",)

    def __init__(
        self,
        providers: Mapping[str, DelegatedExecutionProvider[Any, Any]],
    ) -> None:
        self._providers = dict(providers)

    def resolve(
        self,
        provider_id: str,
    ) -> DelegatedExecutionProvider[Any, Any] | None:
        normalized = provider_id.strip()
        if not normalized:
            return None
        return self._providers.get(normalized)


__all__ = ["MappingDelegatedExecutionProviderResolver"]
