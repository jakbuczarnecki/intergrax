# © Artur Czarnecki. All rights reserved.

"""Delegate resolver with explicit host factory overrides (revision-bound AC3 seam)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolver,
    _factory_key,
)
from intergrax.applications.contracts.factory import CanonicalAgentFactory

_FactoryKey = tuple[str, str | None, str | None]


@dataclass(frozen=True, slots=True)
class HostRuntimeAgentFactoryOverride:
    """AC3 factory keys wrapped after authoritative delegate resolution."""

    factory_keys: frozenset[_FactoryKey]
    wrap_authoritative_factory: Callable[
        [CanonicalAgentFactory],
        CanonicalAgentFactory,
    ]


class HostOverridingRuntimeAgentFactoryResolver:
    """Single resolver authority: delegate all factories except explicit host overrides."""

    __slots__ = ("_delegate", "_override")

    def __init__(
        self,
        delegate: RuntimeAgentFactoryResolver,
        *,
        override: HostRuntimeAgentFactoryOverride,
    ) -> None:
        self._delegate = delegate
        self._override = override

    def resolve_factory(
        self,
        *,
        runtime_revision: RuntimeRevision,
        package_digest: str,
        factory_reference: AgentBindingFactoryReference,
    ) -> CanonicalAgentFactory:
        key = _factory_key(package_digest, factory_reference)
        authoritative = self._delegate.resolve_factory(
            runtime_revision=runtime_revision,
            package_digest=package_digest,
            factory_reference=factory_reference,
        )
        if key in self._override.factory_keys:
            return self._override.wrap_authoritative_factory(authoritative)
        return authoritative


__all__ = [
    "HostOverridingRuntimeAgentFactoryResolver",
    "HostRuntimeAgentFactoryOverride",
]
