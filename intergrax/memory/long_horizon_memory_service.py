# © Artur Czarnecki. All rights reserved.

"""Long-horizon memory capability — compaction, recall, lineage (MEM-ENT-9)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceAuthority,
    ChildSummaryRef,
    CompactionPartialFailure,
    CompactionResult,
    DefaultDeterministicGroupingStrategy,
    DefaultLongHorizonCompactionPolicy,
    DefaultLongHorizonRecallStrategy,
    DefaultLongHorizonSummaryStrategy,
    LineageTraversalRequest,
    LineageTraversalResult,
    LongHorizonCompactionPolicy,
    LongHorizonCompactionRequest,
    LongHorizonCompactionSource,
    LongHorizonMemoryCapability,
    LongHorizonMemoryScope,
    LongHorizonMemoryStore,
    LongHorizonMemoryViolation,
    LongHorizonPolicyConfig,
    LongHorizonRecallQuery,
    LongHorizonRecallResult,
    LongHorizonRecallStrategy,
    LongHorizonSummaryGroupingStrategy,
    LongHorizonSummaryRecord,
    LongHorizonSummaryStrategy,
    MemorySourceRef,
    SourceRevisionResolver,
    SummaryGenerationRequest,
    SummaryNodeKind,
    SummaryStatus,
    long_horizon_batch_identity_key,
    long_horizon_summary_id_for_batch,
    sort_child_summary_refs,
    sort_memory_source_refs,
    validate_canonical_source_snapshot,
)
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
    validate_canonical_governance_source_snapshot,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_disclosure_governance import (
    evaluate_memory_disclosure,
    filter_memory_disclosure_candidates,
    memory_security_context_for_recall,
)
from intergrax.memory.memory_specialized_mutation_governance import (
    enforce_specialized_memory_mutation,
    governance_snapshot_from_long_horizon_summary,
    memory_security_context_for_mutation,
    resolve_governance_source_record_snapshot,
    specialized_mutation_denied_for_canonical_source,
)

__all__ = [
    "LongHorizonMemoryService",
    "LongHorizonMemoryStrategySet",
    "build_default_long_horizon_strategies",
]


@dataclass(frozen=True, slots=True)
class LongHorizonMemoryStrategySet:
    grouping: LongHorizonSummaryGroupingStrategy
    summary: LongHorizonSummaryStrategy
    compaction: LongHorizonCompactionPolicy
    recall: LongHorizonRecallStrategy


def build_default_long_horizon_strategies() -> LongHorizonMemoryStrategySet:
    return LongHorizonMemoryStrategySet(
        grouping=DefaultDeterministicGroupingStrategy(),
        summary=DefaultLongHorizonSummaryStrategy(),
        compaction=DefaultLongHorizonCompactionPolicy(),
        recall=DefaultLongHorizonRecallStrategy(),
    )


def _validate_child_levels(
    parent_level: int,
    children: tuple[LongHorizonSummaryRecord, ...],
) -> None:
    for child in children:
        if parent_level <= child.summary_level:
            raise LongHorizonMemoryViolation(
                "parent summary level must be greater than each child level"
            )


def _validate_children_scope(
    scope: LongHorizonMemoryScope,
    store: LongHorizonMemoryStore,
    children: tuple[LongHorizonSummaryRecord, ...],
) -> None:
    for child in children:
        stored = store.get_summary(scope, child.summary_id)
        if stored is None:
            raise LongHorizonMemoryViolation(f"child summary not found: {child.summary_id}")
        if stored.revision != child.revision:
            raise LongHorizonMemoryViolation(
                f"child summary revision mismatch for {child.summary_id}"
            )


@dataclass(slots=True)
class LongHorizonMemoryService:
    """Default ``LongHorizonMemoryCapability`` backed by ``LongHorizonMemoryStore``."""

    _store: LongHorizonMemoryStore
    _strategies: LongHorizonMemoryStrategySet
    _source_authority: CanonicalMemorySourceAuthority
    _governance_source_authority: CanonicalMemoryGovernanceSourceAuthority
    _security_governance: MemorySecurityGovernanceService
    _policy: LongHorizonPolicyConfig = LongHorizonPolicyConfig()

    def compact(self, request: LongHorizonCompactionRequest) -> CompactionResult:
        if request.target_level > self._policy.max_hierarchy_depth:
            raise LongHorizonMemoryViolation("target_level exceeds max_hierarchy_depth")

        created: list[LongHorizonSummaryRecord] = []
        updated: list[LongHorizonSummaryRecord] = []
        skipped: list[str] = []
        failures: list[CompactionPartialFailure] = []

        if request.sources:
            if not self._strategies.compaction.should_compact_sources(
                request.sources,
                policy=self._policy,
            ):
                return CompactionResult(skipped_batch_keys=("sources-not-ready",))
            batches = self._strategies.grouping.group_sources(
                request.sources,
                max_batch=self._policy.max_source_batch,
            )
            node_kind = SummaryNodeKind.LEAF
            leaf_operation = MemoryGovernanceOperation.COMPACT
            for batch in batches:
                try:
                    canonical_batch = self._resolve_canonical_source_batch(request.scope, batch)
                    for source in canonical_batch:
                        self._assert_governance_matches_content_authority(
                            request.scope,
                            source,
                            leaf_operation,
                        )
                    record, skipped_idempotent = self._materialize_summary(
                        scope=request.scope,
                        target_level=request.target_level,
                        node_kind=node_kind,
                        sources=canonical_batch,
                        children=(),
                        reference_time=request.reference_time,
                    )
                    if skipped_idempotent:
                        skipped.append(record.summary_id)
                        continue
                    self._enforce_compaction_mutation(
                        request.identity,
                        request.scope,
                        record,
                        node_kind=node_kind,
                        source_records=self._source_governance_snapshots(
                            request.scope,
                            canonical_batch,
                            (),
                            operation=leaf_operation,
                        ),
                    )
                    outcome = self._persist_summary(request.scope, record)
                    if outcome == "created":
                        created.append(record)
                    else:
                        updated.append(record)
                except (LongHorizonMemoryViolation, MemoryGovernanceDenied) as exc:
                    batch_key = long_horizon_batch_identity_key(
                        source_refs=tuple(
                            MemorySourceRef(memory_id=s.memory_id, revision=s.revision)
                            for s in batch
                        )
                    )
                    failures.append(CompactionPartialFailure(batch_identity=batch_key, message=str(exc)))
        else:
            children = request.child_summaries
            if not self._strategies.compaction.should_promote_children(
                children,
                policy=self._policy,
            ):
                return CompactionResult(skipped_batch_keys=("children-not-ready",))
            _validate_child_levels(request.target_level, children)
            batches = self._strategies.grouping.group_child_summaries(
                children,
                max_batch=self._policy.max_source_batch,
            )
            node_kind = SummaryNodeKind.AGGREGATE
            promote_operation = MemoryGovernanceOperation.PROMOTE
            for batch in batches:
                try:
                    _validate_children_scope(request.scope, self._store, batch)
                    record, skipped_idempotent = self._materialize_summary(
                        scope=request.scope,
                        target_level=request.target_level,
                        node_kind=node_kind,
                        sources=(),
                        children=batch,
                        reference_time=request.reference_time,
                    )
                    if skipped_idempotent:
                        skipped.append(record.summary_id)
                        continue
                    self._enforce_compaction_mutation(
                        request.identity,
                        request.scope,
                        record,
                        node_kind=node_kind,
                        source_records=self._source_governance_snapshots(
                            request.scope,
                            (),
                            batch,
                            operation=promote_operation,
                        ),
                    )
                    outcome = self._persist_summary(request.scope, record)
                    if outcome == "created":
                        created.append(record)
                    else:
                        updated.append(record)
                except (LongHorizonMemoryViolation, MemoryGovernanceDenied) as exc:
                    batch_key = long_horizon_batch_identity_key(
                        child_refs=tuple(
                            ChildSummaryRef(summary_id=c.summary_id, revision=c.revision)
                            for c in batch
                        )
                    )
                    failures.append(CompactionPartialFailure(batch_identity=batch_key, message=str(exc)))

        return CompactionResult(
            created=tuple(created),
            updated=tuple(updated),
            skipped_batch_keys=tuple(skipped),
            failures=tuple(failures),
        )

    def _enforce_compaction_mutation(
        self,
        identity: RequestIdentity,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
        *,
        node_kind: SummaryNodeKind,
        source_records: tuple[MemoryGovernanceRecordSnapshot, ...],
    ) -> None:
        operation = (
            MemoryGovernanceOperation.PROMOTE
            if node_kind is SummaryNodeKind.AGGREGATE
            else MemoryGovernanceOperation.COMPACT
        )
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                proposed_record=governance_snapshot_from_long_horizon_summary(record),
                source_records=source_records,
            ),
        )

    def _source_governance_snapshots(
        self,
        scope: LongHorizonMemoryScope,
        sources: tuple[LongHorizonCompactionSource, ...],
        children: tuple[LongHorizonSummaryRecord, ...],
        *,
        operation: MemoryGovernanceOperation,
    ) -> tuple[MemoryGovernanceRecordSnapshot, ...]:
        snapshots: list[MemoryGovernanceRecordSnapshot] = []
        for child in children:
            snapshots.append(governance_snapshot_from_long_horizon_summary(child))
        if children:
            return tuple(snapshots)
        for source in sources:
            preview = source.content[:256] if source.content else None
            snapshots.append(
                resolve_governance_source_record_snapshot(
                    self._governance_source_authority,
                    scope,
                    source.memory_id,
                    source.revision,
                    operation=operation,
                    content_preview=preview,
                )
            )
        return tuple(snapshots)

    def _assert_governance_matches_content_authority(
        self,
        scope: LongHorizonMemoryScope,
        source: LongHorizonCompactionSource,
        operation: MemoryGovernanceOperation,
    ) -> None:
        try:
            governance = self._governance_source_authority.resolve_canonical_governance_source(
                scope,
                source.memory_id,
                source.revision,
            )
        except Exception as exc:
            raise specialized_mutation_denied_for_canonical_source(operation, str(exc)) from exc
        validate_canonical_governance_source_snapshot(
            scope,
            source.memory_id,
            source.revision,
            governance,
        )
        content = self._source_authority.resolve_canonical_source(
            scope,
            source.memory_id,
            source.revision,
        )
        validate_canonical_source_snapshot(source, content)
        if content.memory_id != governance.memory_id or content.revision != governance.revision:
            raise specialized_mutation_denied_for_canonical_source(
                operation,
                "canonical content and governance authorities disagree on memory identity",
            )

    def _resolve_canonical_source_batch(
        self,
        scope: LongHorizonMemoryScope,
        batch: tuple[LongHorizonCompactionSource, ...],
    ) -> tuple[LongHorizonCompactionSource, ...]:
        resolved: list[LongHorizonCompactionSource] = []
        for source in batch:
            snapshot = self._source_authority.resolve_canonical_source(
                scope,
                source.memory_id,
                source.revision,
            )
            validate_canonical_source_snapshot(source, snapshot)
            resolved.append(
                LongHorizonCompactionSource(
                    memory_id=snapshot.memory_id,
                    revision=snapshot.revision,
                    content=snapshot.content,
                    observed_at=snapshot.observed_at,
                )
            )
        return tuple(resolved)

    def _materialize_summary(
        self,
        *,
        scope: LongHorizonMemoryScope,
        target_level: int,
        node_kind: SummaryNodeKind,
        sources: tuple[LongHorizonCompactionSource, ...],
        children: tuple[LongHorizonSummaryRecord, ...],
        reference_time,
    ) -> tuple[LongHorizonSummaryRecord, bool]:
        generation = self._strategies.summary.generate(
            SummaryGenerationRequest(
                scope=scope,
                target_level=target_level,
                node_kind=node_kind,
                sources=sources,
                child_summaries=children,
                reference_time=reference_time,
            )
        )
        if node_kind is SummaryNodeKind.LEAF:
            batch_key = long_horizon_batch_identity_key(source_refs=generation.source_memory_refs)
            source_count = len(generation.source_memory_refs)
        else:
            batch_key = long_horizon_batch_identity_key(child_refs=generation.child_summary_refs)
            source_count = sum(child.source_count or len(child.source_memory_refs) for child in children)

        summary_id = long_horizon_summary_id_for_batch(scope, target_level, batch_key)
        existing = self._store.get_summary(scope, summary_id)
        source_refs = sort_memory_source_refs(generation.source_memory_refs)
        child_refs = sort_child_summary_refs(generation.child_summary_refs)
        if existing is not None:
            if (
                existing.source_memory_refs == source_refs
                and existing.child_summary_refs == child_refs
                and existing.summary_level == target_level
                and existing.node_kind is node_kind
            ):
                return existing, True

        revision = 1 if existing is None else existing.revision + 1
        timestamp = ""
        if reference_time is not None:
            timestamp = reference_time.isoformat()

        record = LongHorizonSummaryRecord(
            summary_id=summary_id,
            summary_level=target_level,
            node_kind=node_kind,
            revision=revision,
            content=generation.content,
            source_memory_refs=source_refs,
            child_summary_refs=child_refs,
            covered_from=generation.covered_from,
            covered_until=generation.covered_until,
            source_count=source_count,
            evidence_refs=generation.evidence_refs,
            created_at=existing.created_at if existing and existing.created_at else timestamp,
            updated_at=timestamp or None,
            status=SummaryStatus.ACTIVE,
        )
        return record, False

    def _persist_summary(
        self,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
    ) -> str:
        existing = self._store.get_summary(scope, record.summary_id)
        if existing is None:
            self._store.upsert_summary(scope, record)
            return "created"
        if existing.revision == record.revision:
            self._store.upsert_summary(scope, record)
            return "updated"
        if record.revision > existing.revision:
            self._store.upsert_summary(scope, record)
            return "updated"
        raise LongHorizonMemoryViolation("summary revision regression during compaction")

    def _summary_permits_disclosure(
        self,
        identity: RequestIdentity,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
    ) -> bool:
        context = memory_security_context_for_recall(identity, scope)
        return evaluate_memory_disclosure(
            self._security_governance,
            context,
            governance_snapshot_from_long_horizon_summary(record),
        )

    def _source_ref_permits_disclosure(
        self,
        identity: RequestIdentity,
        scope: LongHorizonMemoryScope,
        ref: MemorySourceRef,
    ) -> bool:
        try:
            snapshot = resolve_governance_source_record_snapshot(
                self._governance_source_authority,
                scope,
                ref.memory_id,
                ref.revision,
                operation=MemoryGovernanceOperation.RECALL,
            )
        except MemoryGovernanceDenied:
            return False
        context = memory_security_context_for_recall(identity, scope)
        return evaluate_memory_disclosure(self._security_governance, context, snapshot)

    def recall(
        self,
        identity: RequestIdentity,
        scope: LongHorizonMemoryScope,
        query: LongHorizonRecallQuery,
    ) -> LongHorizonRecallResult:
        bounded_query = LongHorizonRecallQuery(
            levels=query.levels,
            statuses=query.statuses,
            covered_from=query.covered_from,
            covered_until=query.covered_until,
            as_of=query.as_of,
            limit=query.limit,
        )
        candidates = self._store.query_summaries(scope, bounded_query)
        recall_context = memory_security_context_for_recall(identity, scope)
        disclosed = filter_memory_disclosure_candidates(
            self._security_governance,
            recall_context,
            candidates,
            to_snapshot=governance_snapshot_from_long_horizon_summary,
        )
        ranked = self._strategies.recall.rank(disclosed, bounded_query)
        return LongHorizonRecallResult(summaries=ranked[: bounded_query.limit])

    def traverse_lineage(
        self,
        identity: RequestIdentity,
        scope: LongHorizonMemoryScope,
        request: LineageTraversalRequest,
    ) -> LineageTraversalResult:
        root = self._store.get_summary(scope, request.summary_id)
        if root is None:
            raise LongHorizonMemoryViolation("summary not found for lineage traversal")
        if not self._summary_permits_disclosure(identity, scope, root):
            return LineageTraversalResult(
                visited_summaries=(),
                canonical_source_refs=(),
                truncated=False,
                cycle_detected=False,
            )

        visited_summaries: list[LongHorizonSummaryRecord] = []
        canonical_refs: list[MemorySourceRef] = []
        visited_ids: set[str] = set()
        truncated = False
        cycle_detected = False

        def visit(record: LongHorizonSummaryRecord, depth: int) -> None:
            nonlocal truncated, cycle_detected
            if len(visited_summaries) >= request.max_nodes:
                truncated = True
                return
            if depth > request.max_depth:
                truncated = True
                return
            sid = record.summary_id.strip()
            if sid in visited_ids:
                cycle_detected = True
                return
            if not self._summary_permits_disclosure(identity, scope, record):
                return
            visited_ids.add(sid)
            visited_summaries.append(record)

            if record.node_kind is SummaryNodeKind.LEAF:
                for ref in record.source_memory_refs:
                    if len(canonical_refs) >= request.max_nodes:
                        truncated = True
                        return
                    if self._source_ref_permits_disclosure(identity, scope, ref):
                        canonical_refs.append(ref)
                return

            for child_ref in record.child_summary_refs:
                if len(visited_summaries) >= request.max_nodes:
                    truncated = True
                    return
                child = self._store.get_summary(scope, child_ref.summary_id)
                if child is None:
                    raise LongHorizonMemoryViolation(
                        f"missing child summary during traversal: {child_ref.summary_id}"
                    )
                if child.revision != child_ref.revision:
                    raise LongHorizonMemoryViolation(
                        f"child revision mismatch during traversal: {child_ref.summary_id}"
                    )
                visit(child, depth + 1)

        visit(root, 0)
        if cycle_detected:
            raise LongHorizonMemoryViolation("cycle detected in summary hierarchy")

        return LineageTraversalResult(
            visited_summaries=tuple(visited_summaries),
            canonical_source_refs=sort_memory_source_refs(tuple(canonical_refs)),
            truncated=truncated,
            cycle_detected=False,
        )

    def validate_summary_sources(
        self,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
        resolver: SourceRevisionResolver,
    ) -> SummaryStatus:
        if record.node_kind is SummaryNodeKind.AGGREGATE:
            for child_ref in record.child_summary_refs:
                child = self._store.get_summary(scope, child_ref.summary_id)
                if child is None:
                    return SummaryStatus.INVALIDATED
                child_status = self.validate_summary_sources(scope, child, resolver)
                if child_status is not SummaryStatus.ACTIVE:
                    return SummaryStatus.STALE
            return record.status

        for ref in record.source_memory_refs:
            current = resolver.resolve_source_revision(scope, ref.memory_id)
            if current is None:
                return SummaryStatus.INVALIDATED
            if current != ref.revision:
                return SummaryStatus.STALE
        return record.status

    def invalidate_summaries_for_deleted_source(
        self,
        identity: RequestIdentity,
        scope: LongHorizonMemoryScope,
        source_memory_id: str,
    ) -> tuple[str, ...]:
        memory_id = (source_memory_id or "").strip()
        if not memory_id:
            return ()
        query = LongHorizonRecallQuery(
            statuses=(
                SummaryStatus.ACTIVE,
                SummaryStatus.STALE,
                SummaryStatus.SUPERSEDED,
            ),
            limit=500,
        )
        candidates = self._store.query_summaries(scope, query)
        invalidated: list[str] = []
        for record in candidates:
            if record.node_kind is SummaryNodeKind.LEAF:
                if any(ref.memory_id == memory_id for ref in record.source_memory_refs):
                    enforce_specialized_memory_mutation(
                        self._security_governance,
                        MemoryGovernanceEvaluationRequest(
                            context=memory_security_context_for_mutation(
                                identity, scope, MemoryGovernanceOperation.DELETE
                            ),
                            target=MemoryGovernanceTarget(memory_id=record.summary_id),
                            existing_record=governance_snapshot_from_long_horizon_summary(record),
                        ),
                    )
                    updated = self._store.invalidate_summary(scope, record.summary_id)
                    if updated is not None:
                        invalidated.append(record.summary_id)
        return tuple(invalidated)
