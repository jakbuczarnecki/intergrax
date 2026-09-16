# © Artur Czarnecki. All rights reserved.

"""Canonical procedural memory contracts (MEM-ENT-8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordTrust,
    parse_memory_record_timestamp,
)
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    entity_memory_source_projection_key,
)
from intergrax.memory.contracts.temporal_chronology import memory_timestamps_same_awareness

ProceduralMemoryScope = EntityMemoryScope

__all__ = [
    "DefaultProcedureApplicabilityStrategy",
    "DefaultProcedureRankingStrategy",
    "ProcedureActionKind",
    "ProcedureApplicability",
    "ProcedureApplicabilityStrategy",
    "ProcedureMemoryCapability",
    "ProceduralMemoryScope",
    "ProcedureMemoryStore",
    "ProcedureMemoryViolation",
    "ProcedureOutcomeEvidence",
    "ProcedureQuery",
    "ProcedureRankingStrategy",
    "ProcedureRecallContext",
    "ProcedureRecallResult",
    "ProcedureRecord",
    "ProcedureStatus",
    "ProcedureStep",
    "ProcedureSupersessionRequest",
    "ProcedureToolReference",
    "ProcedureTypeRef",
    "is_procedure_temporally_active",
    "order_procedures_deterministic",
    "procedure_id_for_source_memory",
    "procedure_memory_source_projection_key",
    "procedure_status_recall_priority",
]


class ProcedureMemoryViolation(ValueError):
    """Record or query violates procedural memory invariants."""


class ProcedureStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    DISABLED = "disabled"


class ProcedureActionKind(str, Enum):
    TOOL_ACTION = "tool_action"
    MODEL_REASONING = "model_reasoning"
    VALIDATION = "validation"
    HUMAN_APPROVAL = "human_approval"
    SUBPROCESS = "subprocess"


def procedure_status_recall_priority(status: ProcedureStatus) -> int:
    """Lower value sorts earlier when ranking by status."""
    return {
        ProcedureStatus.ACTIVE: 0,
        ProcedureStatus.DEPRECATED: 1,
        ProcedureStatus.SUPERSEDED: 2,
        ProcedureStatus.DISABLED: 3,
    }[status]


@dataclass(frozen=True, slots=True)
class ProcedureTypeRef:
    value: str

    def __post_init__(self) -> None:
        if not (self.value or "").strip():
            raise ProcedureMemoryViolation("procedure_type must be non-empty")


@dataclass(frozen=True, slots=True)
class ProcedureToolReference:
    tool_capability_id: str | None = None
    tool_contract_id: str | None = None

    def __post_init__(self) -> None:
        cap = (self.tool_capability_id or "").strip()
        contract = (self.tool_contract_id or "").strip()
        if not cap and not contract:
            raise ProcedureMemoryViolation(
                "tool reference requires tool_capability_id or tool_contract_id"
            )


@dataclass(frozen=True, slots=True)
class ProcedureStep:
    step_id: str
    position: int
    action_kind: ProcedureActionKind
    instruction: str
    preconditions: tuple[str, ...] = ()
    expected_outcome: str | None = None
    tool_reference: ProcedureToolReference | None = None

    def __post_init__(self) -> None:
        if not (self.step_id or "").strip():
            raise ProcedureMemoryViolation("step_id must be non-empty")
        if self.position < 0:
            raise ProcedureMemoryViolation("position must be >= 0")
        if not (self.instruction or "").strip():
            raise ProcedureMemoryViolation("instruction must be non-empty")


@dataclass(frozen=True, slots=True)
class ProcedureApplicability:
    domains: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    context_tags: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_until: str | None = None

    def __post_init__(self) -> None:
        if self.valid_from and self.valid_until:
            from_dt = parse_memory_record_timestamp("valid_from", self.valid_from)
            until_dt = parse_memory_record_timestamp("valid_until", self.valid_until)
            if not memory_timestamps_same_awareness(from_dt, until_dt):
                raise ProcedureMemoryViolation(
                    "valid_from/valid_until: timezone-aware and naive timestamps are not comparable"
                )
            if from_dt > until_dt:
                raise ProcedureMemoryViolation("valid_from must not be after valid_until")


@dataclass(frozen=True, slots=True)
class ProcedureOutcomeEvidence:
    success_count: int = 0
    failure_count: int = 0
    last_success_at: str | None = None
    quality_score: float | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.success_count < 0 or self.failure_count < 0:
            raise ProcedureMemoryViolation("outcome counts must be >= 0")


@dataclass(frozen=True, slots=True)
class ProcedureRecord:
    procedure_id: str
    procedure_type: ProcedureTypeRef
    title: str
    revision: int = 1
    status: ProcedureStatus = ProcedureStatus.ACTIVE
    steps: tuple[ProcedureStep, ...] = ()
    applicability: ProcedureApplicability = field(default_factory=ProcedureApplicability)
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    trust: MemoryRecordTrust = field(default_factory=MemoryRecordTrust)
    governance: MemoryRecordGovernance = field(default_factory=MemoryRecordGovernance)
    outcome_evidence: ProcedureOutcomeEvidence = field(default_factory=ProcedureOutcomeEvidence)
    evidence_refs: tuple[str, ...] = ()
    source_memory_id: str | None = None
    source_memory_revision: int | None = None
    superseded_by_procedure_id: str | None = None
    created_at: str = ""
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if not (self.procedure_id or "").strip():
            raise ProcedureMemoryViolation("procedure_id must be non-empty")
        if self.revision < 1:
            raise ProcedureMemoryViolation("revision must be >= 1")
        if not (self.title or "").strip():
            raise ProcedureMemoryViolation("title must be non-empty")
        if self.status is ProcedureStatus.SUPERSEDED and not (
            self.superseded_by_procedure_id or ""
        ).strip():
            raise ProcedureMemoryViolation(
                "superseded procedures require superseded_by_procedure_id"
            )


@dataclass(frozen=True, slots=True)
class ProcedureQuery:
    procedure_type: str | None = None
    statuses: tuple[ProcedureStatus, ...] = (ProcedureStatus.ACTIVE,)
    required_capabilities: tuple[str, ...] = ()
    context_tags: tuple[str, ...] = ()
    include_history: bool = False
    as_of: datetime | None = None
    limit: int = 20

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ProcedureMemoryViolation("limit must be >= 1")


@dataclass(frozen=True, slots=True)
class ProcedureRecallContext:
    """Inputs for applicability and ranking (no hidden wall clock)."""

    available_capabilities: tuple[str, ...] = ()
    context_tags: tuple[str, ...] = ()
    domain: str | None = None
    task_type: str | None = None
    reference_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProcedureRecallResult:
    procedures: tuple[ProcedureRecord, ...]


@dataclass(frozen=True, slots=True)
class ProcedureSupersessionRequest:
    superseded_procedure_id: str
    superseding_record: ProcedureRecord


def procedure_memory_source_projection_key(
    scope: ProceduralMemoryScope,
    source_memory_id: str,
) -> str:
    """Reuse canonical memory projection identity."""
    return entity_memory_source_projection_key(scope, source_memory_id)


def procedure_id_for_source_memory(
    scope: ProceduralMemoryScope,
    source_memory_id: str,
) -> str:
    projection_key = procedure_memory_source_projection_key(scope, source_memory_id)
    return f"proc:memory:{projection_key}"


def _applicability_bound_datetimes(
    applicability: ProcedureApplicability,
) -> tuple[datetime | None, datetime | None]:
    from_dt: datetime | None = None
    until_dt: datetime | None = None
    if applicability.valid_from:
        from_dt = parse_memory_record_timestamp("valid_from", applicability.valid_from)
    if applicability.valid_until:
        until_dt = parse_memory_record_timestamp("valid_until", applicability.valid_until)
    if from_dt is not None and until_dt is not None:
        if not memory_timestamps_same_awareness(from_dt, until_dt):
            raise ProcedureMemoryViolation(
                "valid_from/valid_until: timezone-aware and naive timestamps are not comparable"
            )
    return from_dt, until_dt


def is_procedure_temporally_active(
    record: ProcedureRecord,
    *,
    as_of: datetime,
) -> bool:
    """Return True when procedure applicability window includes ``as_of``."""
    from_dt, until_dt = _applicability_bound_datetimes(record.applicability)
    if from_dt is not None:
        if not memory_timestamps_same_awareness(as_of, from_dt):
            raise ProcedureMemoryViolation("as_of vs valid_from: mixed timezone awareness")
        if as_of < from_dt:
            return False
    if until_dt is not None:
        if not memory_timestamps_same_awareness(as_of, until_dt):
            raise ProcedureMemoryViolation("as_of vs valid_until: mixed timezone awareness")
        if as_of >= until_dt:
            return False
    return True


def _procedure_updated_ordinal(record: ProcedureRecord) -> float:
    stamp = record.updated_at or record.created_at
    if not stamp:
        return float("-inf")
    parsed = parse_memory_record_timestamp("updated_at", stamp)
    return -parsed.timestamp()


def order_procedures_deterministic(
    procedures: tuple[ProcedureRecord, ...],
) -> tuple[ProcedureRecord, ...]:
    def sort_key(item: ProcedureRecord) -> tuple[int, float, float, str]:
        status_pri = procedure_status_recall_priority(item.status)
        quality = item.outcome_evidence.quality_score
        quality_key = -(quality if quality is not None else float("-inf"))
        return (status_pri, quality_key, _procedure_updated_ordinal(item), item.procedure_id)

    return tuple(sorted(procedures, key=sort_key))


@runtime_checkable
class ProcedureMemoryStore(Protocol):
    """Pluggable persistence and query for procedural memory projections."""

    def upsert_procedure(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord: ...

    def get_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None: ...

    def query_procedure_candidates(
        self,
        scope: ProceduralMemoryScope,
        query: ProcedureQuery,
    ) -> tuple[ProcedureRecord, ...]: ...

    def deprecate_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None: ...

    def apply_supersession(
        self,
        scope: ProceduralMemoryScope,
        request: ProcedureSupersessionRequest,
    ) -> tuple[ProcedureRecord, ProcedureRecord]: ...

    def delete_by_source_memory(
        self,
        scope: ProceduralMemoryScope,
        source_memory_id: str,
    ) -> int: ...


@runtime_checkable
class ProcedureApplicabilityStrategy(Protocol):
    def is_applicable(
        self,
        record: ProcedureRecord,
        context: ProcedureRecallContext,
        *,
        query: ProcedureQuery,
    ) -> bool: ...


@runtime_checkable
class ProcedureRankingStrategy(Protocol):
    def rank(
        self,
        candidates: tuple[ProcedureRecord, ...],
        context: ProcedureRecallContext,
    ) -> tuple[ProcedureRecord, ...]: ...


class DefaultProcedureApplicabilityStrategy:
    """Deterministic capability, context, domain, and temporal filtering."""

    def is_applicable(
        self,
        record: ProcedureRecord,
        context: ProcedureRecallContext,
        *,
        query: ProcedureQuery,
    ) -> bool:
        if not query.include_history:
            if record.status is not ProcedureStatus.ACTIVE:
                return False
        elif record.status not in query.statuses:
            return False

        required = {cap.strip() for cap in record.applicability.required_capabilities if cap.strip()}
        available = {cap.strip() for cap in context.available_capabilities if cap.strip()}
        if required and not required.issubset(available):
            return False

        if query.required_capabilities:
            query_caps = {cap.strip() for cap in query.required_capabilities if cap.strip()}
            if query_caps and not query_caps.issubset(available):
                return False

        record_tags = {tag.strip() for tag in record.applicability.context_tags if tag.strip()}
        if record_tags:
            ctx_tags = {tag.strip() for tag in context.context_tags if tag.strip()}
            if not record_tags.issubset(ctx_tags):
                return False

        if query.context_tags:
            q_tags = {tag.strip() for tag in query.context_tags if tag.strip()}
            ctx_tags = {tag.strip() for tag in context.context_tags if tag.strip()}
            if q_tags and not q_tags.issubset(ctx_tags):
                return False

        if context.domain:
            domains = {d.strip() for d in record.applicability.domains if d.strip()}
            if domains and context.domain.strip() not in domains:
                return False

        if context.task_type:
            types = {t.strip() for t in record.applicability.task_types if t.strip()}
            if types and context.task_type.strip() not in types:
                return False

        as_of = query.as_of if query.as_of is not None else context.reference_time
        if as_of is not None:
            if not is_procedure_temporally_active(record, as_of=as_of):
                return False
        return True


class DefaultProcedureRankingStrategy:
    """Deterministic ranking with ``procedure_id`` ascending final tie-break."""

    def rank(
        self,
        candidates: tuple[ProcedureRecord, ...],
        context: ProcedureRecallContext,
    ) -> tuple[ProcedureRecord, ...]:
        _ = context
        return order_procedures_deterministic(candidates)


@runtime_checkable
class ProcedureMemoryCapability(Protocol):
    """Typed procedural memory surface (projection over canonical memory)."""

    def remember_procedure(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord: ...

    def recall_procedures(
        self,
        scope: ProceduralMemoryScope,
        query: ProcedureQuery,
        context: ProcedureRecallContext,
    ) -> ProcedureRecallResult: ...

    def deprecate_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None: ...

    def supersede_procedure(
        self,
        scope: ProceduralMemoryScope,
        request: ProcedureSupersessionRequest,
    ) -> tuple[ProcedureRecord, ProcedureRecord]: ...

    def delete_projection_by_source_memory(
        self,
        scope: ProceduralMemoryScope,
        source_memory_id: str,
    ) -> int: ...
