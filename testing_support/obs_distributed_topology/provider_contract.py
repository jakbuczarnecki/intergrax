# © Artur Czarnecki. All rights reserved.

"""Provider-neutral qualification contracts for OBS-DG005 (no vendor imports)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from intergrax.contracts.execution_evidence.persistence_port import (
    EvidencePersistencePort,
)
from intergrax.knowledge.contracts.validation import JsonValue


@dataclass(frozen=True, slots=True)
class EvidenceProviderDescriptor:
    """Serializable durable evidence backend selection (worker IPC / scenario.json)."""

    provider_id: str
    config: Mapping[str, JsonValue]


EvidenceProviderFactory = Callable[
    [EvidenceProviderDescriptor],
    EvidencePersistencePort,
]


class QualificationEvidenceProviderResolver:
    """Explicit DI resolver — the only place provider_id branching is allowed."""

    __slots__ = ("_providers",)

    def __init__(
        self,
        *,
        providers: Mapping[str, EvidenceProviderFactory],
    ) -> None:
        if not providers:
            raise ValueError(
                "qualification provider resolver requires at least one factory"
            )
        self._providers = dict(providers)

    def resolve(
        self, descriptor: EvidenceProviderDescriptor
    ) -> EvidencePersistencePort:
        factory = self._providers.get(descriptor.provider_id)
        if factory is None:
            raise ValueError(
                f"unknown qualification evidence provider: {descriptor.provider_id!r}",
            )
        port = factory(descriptor)
        if not isinstance(port, EvidencePersistencePort):
            raise TypeError(
                "qualification evidence provider factory must return EvidencePersistencePort",
            )
        return port
