# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persisted diagnostic Problem record port surface (DIAG-5D / HARDENING-8)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.diagnostics.problem_identity import (
    ProblemId,
    ProblemOccurrenceAggregateHealth,
    ProblemStatus,
)
from intergrax.contracts.diagnostics.reconciliation_key import ProblemReconciliationKey


@runtime_checkable
class ProblemLifecycleProvenance(Protocol):
    """Audit trail for which strategy established or last updated a Problem."""

    @property
    def strategy_id(self) -> str: ...

    @property
    def strategy_version(self) -> str: ...

    @property
    def method(self) -> str: ...

    @property
    def reconciliation_key(self) -> ProblemReconciliationKey: ...


@runtime_checkable
class PersistedProblem(Protocol):
    """
    Durable derived diagnostic Problem row — not canonical execution truth.

    Runtime ``Problem`` dataclass satisfies this port; adapters encode/decode
    without coupling the contract layer to storage vendors.
    """

    @property
    def problem_id(self) -> ProblemId: ...

    @property
    def tenant_id(self) -> str: ...

    @property
    def status(self) -> ProblemStatus: ...

    @property
    def first_seen_at(self) -> datetime: ...

    @property
    def last_seen_at(self) -> datetime: ...

    @property
    def occurrence_count(self) -> int: ...

    @property
    def provenance(self) -> ProblemLifecycleProvenance: ...

    @property
    def record_version(self) -> int: ...

    @property
    def occurrence_aggregate_health(self) -> ProblemOccurrenceAggregateHealth: ...


__all__ = ["PersistedProblem", "ProblemLifecycleProvenance"]
