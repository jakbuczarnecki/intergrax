# © Artur Czarnecki. All rights reserved.

"""Pluggable enterprise evolution operations contracts (DS-E2E-15J-L14)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    AdaptationOperationalReference,
    EvolutionHealthObservation,
    EvolutionOperationRequest,
    EvolutionOperationStatus,
    EvolutionOperationsAuditMetadata,
)


@dataclass(frozen=True, slots=True)
class EvolutionOperationOutcome:
    operational_status: EvolutionOperationStatus
    summary: str


class EnterpriseEvolutionOperationsProvider(Protocol):
    """Pluggable operational control — observes and administers, never creates adaptations."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def operate(
        self, request: EvolutionOperationRequest
    ) -> EvolutionOperationOutcome: ...

    def current_status(
        self, adaptation_reference: AdaptationOperationalReference
    ) -> EvolutionOperationStatus: ...


class EvolutionHealthObservationProvider(Protocol):
    """Supplies health observations — descriptive only, no decisions."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def observe(
        self,
        adaptation_reference: AdaptationOperationalReference,
        *,
        operational_status: EvolutionOperationStatus,
    ) -> EvolutionHealthObservation: ...


class EvolutionOperationsAuditProvider(Protocol):
    """Records auditable trace for every evolution operation."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def build_audit(
        self,
        request: EvolutionOperationRequest,
        *,
        outcome_status: EvolutionOperationStatus,
        operations_provider_id: str,
        operations_provider_version: str,
        health_provider_id: str,
        health_provider_version: str,
        executed_at: datetime,
        outcome_summary: str,
    ) -> EvolutionOperationsAuditMetadata: ...


__all__ = [
    "EnterpriseEvolutionOperationsProvider",
    "EvolutionHealthObservationProvider",
    "EvolutionOperationOutcome",
    "EvolutionOperationsAuditProvider",
]
