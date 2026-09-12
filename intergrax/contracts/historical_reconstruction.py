# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical historical reconstruction coordinates (NPSC-5F/R4, TRACE-ASOF + TRACE-BITEMP)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.bitemporal_knowledge import (
    BitemporalKnowledgeBasis,
    CrossScopeKnowledgeOrderError,
    KnowledgeOrderingScope,
    KnowledgeRevisionWatermark,
    SystemTimeBasis,
    SystemTimeBoundKind,
    ValidTimeBasis,
    ValidTimeBoundKind,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.contracts.execution_event_position import AsOfBoundary


class HistoricalReconstructionError(Exception):
    """Base error for canonical historical reconstruction."""


class HistoricalScopeMismatchError(HistoricalReconstructionError, ValueError):
    """Tenant, run, or knowledge scope does not match the request."""


class KnowledgeBoundaryNotFinalizedError(HistoricalReconstructionError):
    """Requested knowledge watermark exceeds the durable finalized prefix."""


class HistoricalEvidenceIntegrityError(HistoricalReconstructionError):
    """Canonical durable evidence cannot satisfy the requested historical boundary."""


@dataclass(frozen=True, slots=True)
class HistoricalReconstructionBasis:
    """Exact E + K + bitemporal query axes used to produce a historical view."""

    tenant_id: str
    run_id: RunId
    execution_as_of: AsOfBoundary
    knowledge_watermark: KnowledgeRevisionWatermark
    bitemporal_query: BitemporalKnowledgeBasis

    def __post_init__(self) -> None:
        if type(self.tenant_id) is not str:
            raise TypeError("HistoricalReconstructionBasis.tenant_id must be str")
        if not self.tenant_id or self.tenant_id != self.tenant_id.strip():
            raise ValueError(
                "HistoricalReconstructionBasis.tenant_id must be non-empty without surrounding whitespace"
            )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        if type(self.execution_as_of) is not AsOfBoundary:
            raise TypeError("HistoricalReconstructionBasis.execution_as_of must be AsOfBoundary")
        if self.execution_as_of.run_id != self.run_id:
            raise HistoricalScopeMismatchError(
                "execution_as_of.run_id must match HistoricalReconstructionBasis.run_id"
            )
        if type(self.knowledge_watermark) is not KnowledgeRevisionWatermark:
            raise TypeError(
                "HistoricalReconstructionBasis.knowledge_watermark must be KnowledgeRevisionWatermark"
            )
        if self.knowledge_watermark.scope.tenant_id != self.tenant_id:
            raise HistoricalScopeMismatchError(
                "knowledge_watermark scope tenant must match request tenant_id"
            )
        if type(self.bitemporal_query) is not BitemporalKnowledgeBasis:
            raise TypeError(
                "HistoricalReconstructionBasis.bitemporal_query must be BitemporalKnowledgeBasis"
            )


@dataclass(frozen=True, slots=True)
class ExecutionHistoricalReconstructionRequest:
    """Read-only historical reconstruction request with explicit tenant, run, E, K, and VT."""

    tenant_id: str
    run_id: RunId
    execution_as_of: AsOfBoundary
    knowledge_watermark: KnowledgeRevisionWatermark
    bitemporal_query: BitemporalKnowledgeBasis

    def __post_init__(self) -> None:
        if type(self.tenant_id) is not str:
            raise TypeError("ExecutionHistoricalReconstructionRequest.tenant_id must be str")
        if not self.tenant_id or self.tenant_id != self.tenant_id.strip():
            raise ValueError(
                "ExecutionHistoricalReconstructionRequest.tenant_id must be non-empty "
                "without surrounding whitespace"
            )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        if type(self.execution_as_of) is not AsOfBoundary:
            raise TypeError(
                "ExecutionHistoricalReconstructionRequest.execution_as_of must be AsOfBoundary"
            )
        if self.execution_as_of.run_id != self.run_id:
            raise HistoricalScopeMismatchError(
                "execution_as_of.run_id must match ExecutionHistoricalReconstructionRequest.run_id"
            )
        if type(self.knowledge_watermark) is not KnowledgeRevisionWatermark:
            raise TypeError(
                "ExecutionHistoricalReconstructionRequest.knowledge_watermark must be "
                "KnowledgeRevisionWatermark"
            )
        if self.knowledge_watermark.scope.tenant_id != self.tenant_id:
            raise HistoricalScopeMismatchError(
                "knowledge_watermark scope tenant must match request tenant_id"
            )
        if type(self.bitemporal_query) is not BitemporalKnowledgeBasis:
            raise TypeError(
                "ExecutionHistoricalReconstructionRequest.bitemporal_query must be "
                "BitemporalKnowledgeBasis"
            )

    def to_basis(self) -> HistoricalReconstructionBasis:
        return HistoricalReconstructionBasis(
            tenant_id=self.tenant_id,
            run_id=self.run_id,
            execution_as_of=self.execution_as_of,
            knowledge_watermark=self.knowledge_watermark,
            bitemporal_query=self.bitemporal_query,
        )


def require_matching_knowledge_scope(
    *,
    tenant_id: str,
    watermark: KnowledgeRevisionWatermark,
) -> None:
    scope = KnowledgeOrderingScope(tenant_id=tenant_id)
    if watermark.scope.tenant_id != scope.tenant_id:
        raise CrossScopeKnowledgeOrderError(
            "knowledge watermark tenant does not match execution reconstruction tenant"
        )


def valid_time_instant_satisfies_basis(*, instant: datetime, basis: ValidTimeBasis) -> bool:
    """Return whether ``instant`` lies within ``basis`` (half-open interval semantics)."""
    if basis.kind is ValidTimeBoundKind.INSTANT:
        return instant == basis.start
    if instant < basis.start:
        return False
    if basis.end is None:
        return True
    return instant < basis.end


def system_time_instant_satisfies_basis(*, instant: datetime, basis: SystemTimeBasis) -> bool:
    """Return whether ``instant`` lies within ``basis`` (half-open interval semantics)."""
    if basis.kind is SystemTimeBoundKind.INSTANT:
        return instant == basis.start
    if instant < basis.start:
        return False
    if basis.end is None:
        return True
    return instant < basis.end


def revision_known_at_system_instant(
    *,
    query_instant: datetime,
    revision_system_time: SystemTimeBasis,
) -> bool:
    """Whether the revision was known to the platform at ``query_instant`` (system time)."""
    if query_instant < revision_system_time.start:
        return False
    if revision_system_time.kind is SystemTimeBoundKind.INSTANT:
        return query_instant >= revision_system_time.start
    if revision_system_time.end is None:
        return True
    return query_instant < revision_system_time.end


def revision_admissible_at_bitemporal_query(
    *,
    query: BitemporalKnowledgeBasis,
    revision: BitemporalKnowledgeBasis,
) -> bool:
    """Valid-time and system-time axes are evaluated independently (not collapsed)."""
    valid_coordinate = _query_valid_time_coordinate(query.valid_time)
    if not valid_time_instant_satisfies_basis(
        instant=valid_coordinate,
        basis=revision.valid_time,
    ):
        return False
    system_coordinate = _query_system_time_coordinate(query.system_time)
    return revision_known_at_system_instant(
        query_instant=system_coordinate,
        revision_system_time=revision.system_time,
    )


def _query_valid_time_coordinate(basis: ValidTimeBasis) -> datetime:
    return basis.start


def _query_system_time_coordinate(basis: SystemTimeBasis) -> datetime:
    return basis.start
