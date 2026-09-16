# © Artur Czarnecki. All rights reserved.

"""Canonical long-horizon memory contracts (MEM-ENT-9)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
    parse_memory_record_timestamp,
)
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.temporal_chronology import (
    memory_chronological_ordinal,
    memory_timestamps_same_awareness,
)

LongHorizonMemoryScope = EntityMemoryScope

__all__ = [
    "CanonicalMemorySourceAuthority",
    "CanonicalMemorySourceSnapshot",
    "ChildSummaryRef",
    "CompactionPartialFailure",
    "CompactionResult",
    "DefaultDeterministicGroupingStrategy",
    "DefaultLongHorizonCompactionPolicy",
    "DefaultLongHorizonRecallStrategy",
    "DefaultLongHorizonSummaryStrategy",
    "LineageTraversalRequest",
    "LineageTraversalResult",
    "LongHorizonCompactionPolicy",
    "LongHorizonCompactionRequest",
    "LongHorizonCompactionSource",
    "LongHorizonMemoryCapability",
    "LongHorizonMemoryScope",
    "LongHorizonMemoryStore",
    "LongHorizonMemoryViolation",
    "LongHorizonPolicyConfig",
    "LongHorizonRecallQuery",
    "LongHorizonRecallResult",
    "LongHorizonRecallStrategy",
    "LongHorizonSummaryGroupingStrategy",
    "LongHorizonSummaryRecord",
    "LongHorizonSummaryStrategy",
    "MemorySourceRef",
    "SourceRevisionResolver",
    "SummaryGenerationRequest",
    "SummaryGenerationResult",
    "SummaryLevel",
    "SummaryNodeKind",
    "SummaryStatus",
    "long_horizon_summary_id_for_batch",
    "order_long_horizon_summaries_deterministic",
    "select_temporal_coverage",
    "sort_memory_source_refs",
    "sort_child_summary_refs",
    "validate_long_horizon_summary_record",
]


class LongHorizonMemoryViolation(ValueError):
    """Record or query violates long-horizon memory invariants."""


class SummaryNodeKind(str, Enum):
    LEAF = "leaf"
    AGGREGATE = "aggregate"


class SummaryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    STALE = "stale"


class SummaryLevel(int, Enum):
    """Typed hierarchy depth for derived summaries (not canonical memory)."""

    SESSION_LOCAL = 1
    WEEK_TOPIC = 2
    MONTH_LONG = 3


def sort_memory_source_refs(
    refs: tuple[MemorySourceRef, ...],
) -> tuple[MemorySourceRef, ...]:
    return tuple(sorted(refs, key=lambda item: (item.memory_id, item.revision)))


def sort_child_summary_refs(
    refs: tuple[ChildSummaryRef, ...],
) -> tuple[ChildSummaryRef, ...]:
    return tuple(sorted(refs, key=lambda item: (item.summary_id, item.revision)))


@dataclass(frozen=True, slots=True)
class MemorySourceRef:
    memory_id: str
    revision: int

    def __post_init__(self) -> None:
        if not (self.memory_id or "").strip():
            raise LongHorizonMemoryViolation("memory_id must be non-empty")
        if self.revision < 1:
            raise LongHorizonMemoryViolation("source revision must be >= 1")


@dataclass(frozen=True, slots=True)
class ChildSummaryRef:
    summary_id: str
    revision: int

    def __post_init__(self) -> None:
        if not (self.summary_id or "").strip():
            raise LongHorizonMemoryViolation("summary_id must be non-empty")
        if self.revision < 1:
            raise LongHorizonMemoryViolation("child summary revision must be >= 1")


@dataclass(frozen=True, slots=True)
class LongHorizonSummaryRecord:
    summary_id: str
    summary_level: int
    node_kind: SummaryNodeKind
    revision: int
    content: str
    source_memory_refs: tuple[MemorySourceRef, ...] = ()
    child_summary_refs: tuple[ChildSummaryRef, ...] = ()
    covered_from: str | None = None
    covered_until: str | None = None
    source_count: int = 0
    provenance: MemoryProvenance = field(
        default_factory=lambda: MemoryProvenance(
            source_type=MemoryRecordSourceType.SYSTEM,
            strategy_id="long_horizon.compaction",
        )
    )
    trust: MemoryRecordTrust = field(
        default_factory=lambda: MemoryRecordTrust(
            trust_class=MemoryTrustClass.SYSTEM_GENERATED,
        )
    )
    governance: MemoryRecordGovernance = field(default_factory=MemoryRecordGovernance)
    evidence_refs: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str | None = None
    status: SummaryStatus = SummaryStatus.ACTIVE

    def __post_init__(self) -> None:
        validate_long_horizon_summary_record(self)


def _parse_long_horizon_timestamp(field_name: str, value: str) -> datetime:
    try:
        return parse_memory_record_timestamp(field_name, value)
    except ValueError as exc:
        raise LongHorizonMemoryViolation(str(exc)) from exc


def _validate_no_duplicate_memory_refs(refs: tuple[MemorySourceRef, ...]) -> None:
    seen: set[tuple[str, int]] = set()
    for ref in refs:
        key = (ref.memory_id.strip(), ref.revision)
        if key in seen:
            raise LongHorizonMemoryViolation(f"duplicate source memory ref: {ref.memory_id}")
        seen.add(key)


def _validate_no_duplicate_child_refs(refs: tuple[ChildSummaryRef, ...]) -> None:
    seen: set[tuple[str, int]] = set()
    for ref in refs:
        key = (ref.summary_id.strip(), ref.revision)
        if key in seen:
            raise LongHorizonMemoryViolation(f"duplicate child summary ref: {ref.summary_id}")
        seen.add(key)


def _validate_temporal_coverage(covered_from: str | None, covered_until: str | None) -> None:
    if covered_from is not None:
        _parse_long_horizon_timestamp("covered_from", covered_from)
    if covered_until is not None:
        _parse_long_horizon_timestamp("covered_until", covered_until)
    if covered_from is None or covered_until is None:
        return
    from_dt = _parse_long_horizon_timestamp("covered_from", covered_from)
    until_dt = _parse_long_horizon_timestamp("covered_until", covered_until)
    if not memory_timestamps_same_awareness(from_dt, until_dt):
        raise LongHorizonMemoryViolation(
            "covered_from/covered_until: timezone-aware and naive timestamps are not comparable"
        )
    if from_dt > until_dt:
        raise LongHorizonMemoryViolation("covered_from must not be after covered_until")


def validate_long_horizon_summary_record(record: LongHorizonSummaryRecord) -> None:
    if not (record.summary_id or "").strip():
        raise LongHorizonMemoryViolation("summary_id must be non-empty")
    if record.summary_level < 1:
        raise LongHorizonMemoryViolation("summary_level must be >= 1")
    if record.revision < 1:
        raise LongHorizonMemoryViolation("revision must be >= 1")
    if not (record.content or "").strip():
        raise LongHorizonMemoryViolation("summary content must be non-empty")
    if record.node_kind is SummaryNodeKind.LEAF:
        if not record.source_memory_refs:
            raise LongHorizonMemoryViolation("leaf summary requires source_memory_refs")
        if record.child_summary_refs:
            raise LongHorizonMemoryViolation("leaf summary must not have child_summary_refs")
    elif record.node_kind is SummaryNodeKind.AGGREGATE:
        if not record.child_summary_refs:
            raise LongHorizonMemoryViolation("aggregate summary requires child_summary_refs")
        if record.source_memory_refs:
            raise LongHorizonMemoryViolation("aggregate summary must not have source_memory_refs")
    else:
        raise LongHorizonMemoryViolation(f"unknown node_kind: {record.node_kind}")

    _validate_no_duplicate_memory_refs(record.source_memory_refs)
    _validate_no_duplicate_child_refs(record.child_summary_refs)
    _validate_temporal_coverage(record.covered_from, record.covered_until)

    for child in record.child_summary_refs:
        if child.summary_id.strip() == record.summary_id.strip():
            raise LongHorizonMemoryViolation("summary cannot reference itself as child")

    if record.source_count < 0:
        raise LongHorizonMemoryViolation("source_count must be >= 0")
    if record.created_at:
        parse_memory_record_timestamp("created_at", record.created_at)
    if record.updated_at:
        parse_memory_record_timestamp("updated_at", record.updated_at)


def _long_horizon_length_prefixed_segment(value: str) -> str:
    return f"{len(value)}:{value}"


def _encode_identity_segment(value: str) -> str:
    return _long_horizon_length_prefixed_segment(value)


def _long_horizon_encoded_workspace_qualifier(workspace_id: str | None) -> str:
    if workspace_id is None:
        return "0:"
    stripped = workspace_id.strip()
    if not stripped:
        raise LongHorizonMemoryViolation(
            "workspace_id when set must be non-empty for long-horizon identity"
        )
    return _long_horizon_length_prefixed_segment(stripped)


def _long_horizon_scope_identity(scope: LongHorizonMemoryScope) -> str:
    tenant = (scope.tenant_id or "").strip()
    user = (scope.user_id or "").strip()
    if not tenant or not user:
        raise LongHorizonMemoryViolation("tenant_id and user_id required for summary identity")
    workspace = _long_horizon_encoded_workspace_qualifier(scope.workspace_id)
    return "|".join(
        (
            _encode_identity_segment(tenant),
            _encode_identity_segment(user),
            workspace,
        )
    )


def _encode_memory_source_ref_identity(ref: MemorySourceRef) -> str:
    return "|".join(
        (
            _encode_identity_segment(ref.memory_id),
            _encode_identity_segment(str(ref.revision)),
        )
    )


def _encode_child_summary_ref_identity(ref: ChildSummaryRef) -> str:
    return "|".join(
        (
            _encode_identity_segment(ref.summary_id),
            _encode_identity_segment(str(ref.revision)),
        )
    )


def select_temporal_coverage(
    stamps: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """Earliest and latest stamp strings using canonical memory chronology."""
    if not stamps:
        return None, None
    parsed: list[tuple[str, datetime]] = []
    for stamp in stamps:
        dt = _parse_long_horizon_timestamp("temporal_coverage", stamp)
        parsed.append((stamp, dt))
    anchor = parsed[0][1]
    for _, dt in parsed[1:]:
        if not memory_timestamps_same_awareness(anchor, dt):
            raise LongHorizonMemoryViolation(
                "temporal coverage: mixed timezone awareness"
            )
    earliest = min(parsed, key=lambda item: memory_chronological_ordinal(item[1]))
    latest = max(parsed, key=lambda item: memory_chronological_ordinal(item[1]))
    return earliest[0], latest[0]


def long_horizon_batch_identity_key(
    *,
    source_refs: tuple[MemorySourceRef, ...] = (),
    child_refs: tuple[ChildSummaryRef, ...] = (),
) -> str:
    if source_refs and child_refs:
        raise LongHorizonMemoryViolation("batch identity requires either sources or children, not both")
    if source_refs:
        ordered = sort_memory_source_refs(source_refs)
        return "|".join(_encode_memory_source_ref_identity(item) for item in ordered)
    if child_refs:
        ordered = sort_child_summary_refs(child_refs)
        return "|".join(_encode_child_summary_ref_identity(item) for item in ordered)
    raise LongHorizonMemoryViolation("batch identity requires non-empty lineage")


def long_horizon_summary_id_for_batch(
    scope: LongHorizonMemoryScope,
    summary_level: int,
    batch_identity: str,
) -> str:
    """Stable platform identity from scope, level, and deterministic batch lineage."""
    if summary_level < 1:
        raise LongHorizonMemoryViolation("summary_level must be >= 1")
    identity = (batch_identity or "").strip()
    if not identity:
        raise LongHorizonMemoryViolation("batch_identity must be non-empty")
    scope_identity = _long_horizon_scope_identity(scope)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    level_segment = _encode_identity_segment(str(summary_level))
    return f"lhs:{level_segment}:{scope_identity}:{digest}"


@dataclass(frozen=True, slots=True)
class CanonicalMemorySourceSnapshot:
    """Verified canonical memory source for compaction (scope authority must validate)."""

    memory_id: str
    revision: int
    content: str
    observed_at: str | None = None

    def __post_init__(self) -> None:
        if not (self.memory_id or "").strip():
            raise LongHorizonMemoryViolation("memory_id must be non-empty")
        if self.revision < 1:
            raise LongHorizonMemoryViolation("revision must be >= 1")
        if not (self.content or "").strip():
            raise LongHorizonMemoryViolation("canonical source content must be non-empty")
        if self.observed_at:
            parse_memory_record_timestamp("observed_at", self.observed_at)


@dataclass(frozen=True, slots=True)
class LongHorizonPolicyConfig:
    max_hierarchy_depth: int = 4
    max_source_batch: int = 32
    default_recall_limit: int = 20

    def __post_init__(self) -> None:
        if self.max_hierarchy_depth < 1 or self.max_hierarchy_depth > 32:
            raise LongHorizonMemoryViolation("max_hierarchy_depth must be between 1 and 32")
        if self.max_source_batch < 1 or self.max_source_batch > 256:
            raise LongHorizonMemoryViolation("max_source_batch must be between 1 and 256")
        if self.default_recall_limit < 1 or self.default_recall_limit > 500:
            raise LongHorizonMemoryViolation("default_recall_limit must be between 1 and 500")


@dataclass(frozen=True, slots=True)
class LongHorizonCompactionSource:
    memory_id: str
    revision: int
    content: str
    observed_at: str | None = None

    def __post_init__(self) -> None:
        if not (self.memory_id or "").strip():
            raise LongHorizonMemoryViolation("memory_id must be non-empty")
        if self.revision < 1:
            raise LongHorizonMemoryViolation("revision must be >= 1")
        if not (self.content or "").strip():
            raise LongHorizonMemoryViolation("compaction source content must be non-empty")
        if self.observed_at:
            parse_memory_record_timestamp("observed_at", self.observed_at)


def _validate_canonical_source_snapshot(
    requested: LongHorizonCompactionSource,
    snapshot: CanonicalMemorySourceSnapshot,
) -> None:
    if snapshot.memory_id != requested.memory_id:
        raise LongHorizonMemoryViolation(
            "canonical source authority returned unexpected memory_id"
        )
    if snapshot.revision != requested.revision:
        raise LongHorizonMemoryViolation(
            "canonical source authority returned unexpected revision"
        )


@dataclass(frozen=True, slots=True)
class LongHorizonCompactionRequest:
    scope: LongHorizonMemoryScope
    target_level: int
    sources: tuple[LongHorizonCompactionSource, ...] = ()
    child_summaries: tuple[LongHorizonSummaryRecord, ...] = ()
    reference_time: datetime | None = None

    def __post_init__(self) -> None:
        if self.target_level < 1:
            raise LongHorizonMemoryViolation("target_level must be >= 1")
        has_sources = bool(self.sources)
        has_children = bool(self.child_summaries)
        if has_sources == has_children:
            raise LongHorizonMemoryViolation(
                "compaction request requires exactly one of sources or child_summaries"
            )


@dataclass(frozen=True, slots=True)
class SummaryGenerationRequest:
    scope: LongHorizonMemoryScope
    target_level: int
    node_kind: SummaryNodeKind
    sources: tuple[LongHorizonCompactionSource, ...] = ()
    child_summaries: tuple[LongHorizonSummaryRecord, ...] = ()
    size_budget_chars: int = 8000
    reference_time: datetime | None = None

    def __post_init__(self) -> None:
        if self.target_level < 1:
            raise LongHorizonMemoryViolation("target_level must be >= 1")
        if self.size_budget_chars < 1:
            raise LongHorizonMemoryViolation("size_budget_chars must be >= 1")
        if self.node_kind is SummaryNodeKind.LEAF and not self.sources:
            raise LongHorizonMemoryViolation("leaf generation requires sources")
        if self.node_kind is SummaryNodeKind.AGGREGATE and not self.child_summaries:
            raise LongHorizonMemoryViolation("aggregate generation requires child_summaries")


@dataclass(frozen=True, slots=True)
class SummaryGenerationResult:
    content: str
    source_memory_refs: tuple[MemorySourceRef, ...] = ()
    child_summary_refs: tuple[ChildSummaryRef, ...] = ()
    covered_from: str | None = None
    covered_until: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.content or "").strip():
            raise LongHorizonMemoryViolation("generated summary content must be non-empty")
        if not self.source_memory_refs and not self.child_summary_refs:
            raise LongHorizonMemoryViolation("generation result requires lineage refs")


@dataclass(frozen=True, slots=True)
class CompactionPartialFailure:
    batch_identity: str
    message: str


@dataclass(frozen=True, slots=True)
class CompactionResult:
    created: tuple[LongHorizonSummaryRecord, ...] = ()
    updated: tuple[LongHorizonSummaryRecord, ...] = ()
    marked_stale: tuple[str, ...] = ()
    skipped_batch_keys: tuple[str, ...] = ()
    failures: tuple[CompactionPartialFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class LongHorizonRecallQuery:
    levels: tuple[int, ...] = ()
    statuses: tuple[SummaryStatus, ...] = (SummaryStatus.ACTIVE,)
    covered_from: str | None = None
    covered_until: str | None = None
    as_of: datetime | None = None
    limit: int = 20

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise LongHorizonMemoryViolation("limit must be >= 1")
        from_dt = (
            _parse_long_horizon_timestamp("covered_from", self.covered_from)
            if self.covered_from is not None
            else None
        )
        until_dt = (
            _parse_long_horizon_timestamp("covered_until", self.covered_until)
            if self.covered_until is not None
            else None
        )
        if from_dt is not None and until_dt is not None:
            if not memory_timestamps_same_awareness(from_dt, until_dt):
                raise LongHorizonMemoryViolation(
                    "recall covered_from/covered_until: mixed timezone awareness"
                )
            if from_dt > until_dt:
                raise LongHorizonMemoryViolation(
                    "recall covered_from must not be after covered_until"
                )


@dataclass(frozen=True, slots=True)
class LongHorizonRecallResult:
    summaries: tuple[LongHorizonSummaryRecord, ...]


@dataclass(frozen=True, slots=True)
class LineageTraversalRequest:
    summary_id: str
    max_depth: int = 8
    max_nodes: int = 64

    def __post_init__(self) -> None:
        if not (self.summary_id or "").strip():
            raise LongHorizonMemoryViolation("summary_id must be non-empty")
        if self.max_depth < 0:
            raise LongHorizonMemoryViolation("max_depth must be >= 0")
        if self.max_nodes < 1:
            raise LongHorizonMemoryViolation("max_nodes must be >= 1")


@dataclass(frozen=True, slots=True)
class LineageTraversalResult:
    visited_summaries: tuple[LongHorizonSummaryRecord, ...]
    canonical_source_refs: tuple[MemorySourceRef, ...]
    truncated: bool = False
    cycle_detected: bool = False


def _coverage_ordinal(record: LongHorizonSummaryRecord) -> float:
    if not record.covered_until:
        return float("inf")
    parsed = parse_memory_record_timestamp("covered_until", record.covered_until)
    return -memory_chronological_ordinal(parsed)


def order_long_horizon_summaries_deterministic(
    summaries: tuple[LongHorizonSummaryRecord, ...],
) -> tuple[LongHorizonSummaryRecord, ...]:
    def sort_key(item: LongHorizonSummaryRecord) -> tuple[float, int, str]:
        return (_coverage_ordinal(item), -item.summary_level, item.summary_id)

    return tuple(sorted(summaries, key=sort_key))


@runtime_checkable
class LongHorizonMemoryStore(Protocol):
    """Pluggable persistence for long-horizon derived summaries."""

    def upsert_summary(
        self,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
    ) -> LongHorizonSummaryRecord: ...

    def get_summary(
        self,
        scope: LongHorizonMemoryScope,
        summary_id: str,
    ) -> LongHorizonSummaryRecord | None: ...

    def query_summaries(
        self,
        scope: LongHorizonMemoryScope,
        query: LongHorizonRecallQuery,
    ) -> tuple[LongHorizonSummaryRecord, ...]: ...

    def mark_summary_stale(
        self,
        scope: LongHorizonMemoryScope,
        summary_id: str,
    ) -> LongHorizonSummaryRecord | None: ...

    def invalidate_summary(
        self,
        scope: LongHorizonMemoryScope,
        summary_id: str,
    ) -> LongHorizonSummaryRecord | None: ...


@runtime_checkable
class LongHorizonSummaryStrategy(Protocol):
    def generate(self, request: SummaryGenerationRequest) -> SummaryGenerationResult: ...


@runtime_checkable
class LongHorizonSummaryGroupingStrategy(Protocol):
    def group_sources(
        self,
        sources: tuple[LongHorizonCompactionSource, ...],
        *,
        max_batch: int,
    ) -> tuple[tuple[LongHorizonCompactionSource, ...], ...]: ...

    def group_child_summaries(
        self,
        children: tuple[LongHorizonSummaryRecord, ...],
        *,
        max_batch: int,
    ) -> tuple[tuple[LongHorizonSummaryRecord, ...], ...]: ...


@runtime_checkable
class LongHorizonCompactionPolicy(Protocol):
    def should_compact_sources(
        self,
        sources: tuple[LongHorizonCompactionSource, ...],
        *,
        policy: LongHorizonPolicyConfig,
    ) -> bool: ...

    def should_promote_children(
        self,
        children: tuple[LongHorizonSummaryRecord, ...],
        *,
        policy: LongHorizonPolicyConfig,
    ) -> bool: ...


@runtime_checkable
class LongHorizonRecallStrategy(Protocol):
    def rank(
        self,
        candidates: tuple[LongHorizonSummaryRecord, ...],
        query: LongHorizonRecallQuery,
    ) -> tuple[LongHorizonSummaryRecord, ...]: ...


@runtime_checkable
class SourceRevisionResolver(Protocol):
    def resolve_source_revision(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
    ) -> int | None: ...


@runtime_checkable
class CanonicalMemorySourceAuthority(Protocol):
    """Resolve and verify canonical memory sources for compaction (scope-bound)."""

    def resolve_canonical_source(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemorySourceSnapshot: ...


@runtime_checkable
class LongHorizonMemoryCapability(Protocol):
    def compact(self, request: LongHorizonCompactionRequest) -> CompactionResult: ...

    def recall(
        self,
        scope: LongHorizonMemoryScope,
        query: LongHorizonRecallQuery,
    ) -> LongHorizonRecallResult: ...

    def traverse_lineage(
        self,
        scope: LongHorizonMemoryScope,
        request: LineageTraversalRequest,
    ) -> LineageTraversalResult: ...

    def validate_summary_sources(
        self,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
        resolver: SourceRevisionResolver,
    ) -> SummaryStatus: ...

    def invalidate_summaries_for_deleted_source(
        self,
        scope: LongHorizonMemoryScope,
        source_memory_id: str,
    ) -> tuple[str, ...]: ...


class DefaultDeterministicGroupingStrategy:
    """Deterministic batching by stable source ordering and max batch size."""

    def group_sources(
        self,
        sources: tuple[LongHorizonCompactionSource, ...],
        *,
        max_batch: int,
    ) -> tuple[tuple[LongHorizonCompactionSource, ...], ...]:
        if max_batch < 1:
            raise LongHorizonMemoryViolation("max_batch must be >= 1")
        ordered = tuple(sorted(sources, key=lambda item: (item.memory_id, item.revision)))
        batches: list[tuple[LongHorizonCompactionSource, ...]] = []
        for index in range(0, len(ordered), max_batch):
            batches.append(ordered[index : index + max_batch])
        return tuple(batches)

    def group_child_summaries(
        self,
        children: tuple[LongHorizonSummaryRecord, ...],
        *,
        max_batch: int,
    ) -> tuple[tuple[LongHorizonSummaryRecord, ...], ...]:
        if max_batch < 1:
            raise LongHorizonMemoryViolation("max_batch must be >= 1")
        ordered = tuple(sorted(children, key=lambda item: (item.summary_id, item.revision)))
        batches: list[tuple[LongHorizonSummaryRecord, ...]] = []
        for index in range(0, len(ordered), max_batch):
            batches.append(ordered[index : index + max_batch])
        return tuple(batches)


class DefaultLongHorizonCompactionPolicy:
    def should_compact_sources(
        self,
        sources: tuple[LongHorizonCompactionSource, ...],
        *,
        policy: LongHorizonPolicyConfig,
    ) -> bool:
        return len(sources) >= 1

    def should_promote_children(
        self,
        children: tuple[LongHorizonSummaryRecord, ...],
        *,
        policy: LongHorizonPolicyConfig,
    ) -> bool:
        return len(children) >= 1


class DefaultLongHorizonSummaryStrategy:
    """Vendor-neutral deterministic summary stub (no LLM)."""

    def generate(self, request: SummaryGenerationRequest) -> SummaryGenerationResult:
        if request.node_kind is SummaryNodeKind.LEAF:
            parts: list[str] = []
            refs: list[MemorySourceRef] = []
            stamps: list[str] = []
            for source in request.sources:
                snippet = source.content.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                parts.append(f"[{source.memory_id}@{source.revision}] {snippet}")
                refs.append(MemorySourceRef(memory_id=source.memory_id, revision=source.revision))
                if source.observed_at:
                    stamps.append(source.observed_at)
            content = "\n".join(parts)
            if len(content) > request.size_budget_chars:
                content = content[: request.size_budget_chars]
            covered_from, covered_until = select_temporal_coverage(tuple(stamps))
            return SummaryGenerationResult(
                content=content,
                source_memory_refs=sort_memory_source_refs(tuple(refs)),
                covered_from=covered_from,
                covered_until=covered_until,
            )

        child_refs = tuple(
            ChildSummaryRef(summary_id=item.summary_id, revision=item.revision)
            for item in request.child_summaries
        )
        parts = [f"<{item.summary_id}@{item.revision}> {item.content.strip()}" for item in request.child_summaries]
        content = "\n".join(parts)
        if len(content) > request.size_budget_chars:
            content = content[: request.size_budget_chars]
        stamps_from: list[str] = []
        stamps_until: list[str] = []
        for item in request.child_summaries:
            if item.covered_from:
                stamps_from.append(item.covered_from)
            if item.covered_until:
                stamps_until.append(item.covered_until)
        covered_from, _ = (
            select_temporal_coverage(tuple(stamps_from)) if stamps_from else (None, None)
        )
        _, covered_until = (
            select_temporal_coverage(tuple(stamps_until)) if stamps_until else (None, None)
        )
        if covered_from is not None and covered_until is not None:
            from_dt = _parse_long_horizon_timestamp("covered_from", covered_from)
            until_dt = _parse_long_horizon_timestamp("covered_until", covered_until)
            if not memory_timestamps_same_awareness(from_dt, until_dt):
                raise LongHorizonMemoryViolation(
                    "child coverage timestamps have mixed timezone awareness"
                )
        return SummaryGenerationResult(
            content=content,
            child_summary_refs=sort_child_summary_refs(child_refs),
            covered_from=covered_from,
            covered_until=covered_until,
        )


class DefaultLongHorizonRecallStrategy:
    def rank(
        self,
        candidates: tuple[LongHorizonSummaryRecord, ...],
        query: LongHorizonRecallQuery,
    ) -> tuple[LongHorizonSummaryRecord, ...]:
        return order_long_horizon_summaries_deterministic(candidates)
