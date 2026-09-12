# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable audit sinks for decision integration (no storage drivers here)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.contracts.decision.integration.metadata import (
    IntegrationAuditProviderMetadata,
)

if TYPE_CHECKING:
    from intergrax.contracts.decision.integration.audit import (
        DecisionIntegrationAuditRecord,
    )


@dataclass(frozen=True, slots=True)
class DecisionIntegrationAuditEnvelope:
    """Audit payload handed to sinks — record plus provider operational metadata."""

    record: DecisionIntegrationAuditRecord
    provider_metadata: IntegrationAuditProviderMetadata

    def __post_init__(self) -> None:
        from intergrax.contracts.decision.integration.audit import (
            DecisionIntegrationAuditRecord as AuditRecord,
        )

        if type(self.record) is not AuditRecord:
            raise TypeError("record must be DecisionIntegrationAuditRecord")
        if type(self.provider_metadata) is not IntegrationAuditProviderMetadata:
            raise TypeError(
                "provider_metadata must be IntegrationAuditProviderMetadata"
            )


@runtime_checkable
class DecisionAuditSink(Protocol):
    """Sink plugin — production deploys a durable implementation via composition."""

    def append(self, envelope: DecisionIntegrationAuditEnvelope) -> None: ...


@dataclass
class InMemoryDecisionAuditSink:
    """Memory-backed sink for tests and local diagnostics."""

    _entries: list[DecisionIntegrationAuditEnvelope] = field(default_factory=list)

    @property
    def entries(self) -> tuple[DecisionIntegrationAuditEnvelope, ...]:
        return tuple(self._entries)

    def append(self, envelope: DecisionIntegrationAuditEnvelope) -> None:
        if type(envelope) is not DecisionIntegrationAuditEnvelope:
            raise TypeError("envelope must be DecisionIntegrationAuditEnvelope")
        self._entries.append(envelope)


__all__ = [
    "DecisionAuditSink",
    "DecisionIntegrationAuditEnvelope",
    "InMemoryDecisionAuditSink",
]
