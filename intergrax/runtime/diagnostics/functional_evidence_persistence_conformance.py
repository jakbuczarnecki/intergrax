# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Conformance harness for functional evidence persistence backends (DIAG-DURABILITY-D1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineArtifactLineageFact,
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PipelineOutputRelationFact,
    PipelineSelectionFact,
    PipelineValidationLinkFact,
    PlatformFunctionalEvidence,
    ScoreSemantics,
    TypedPipelineScore,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryRequest,
    functional_evidence_query_order_key,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference

_BASE_TIME = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


def sample_functional_evidence_scope(
    *,
    tenant_id: str = "tenant-conformance",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
) -> PipelineEvidenceScope:
    return PipelineEvidenceScope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id,
    )


def sample_functional_evidence(
    *,
    evidence_id: EventId | None = None,
    kind: PipelineEvidenceKind = PipelineEvidenceKind.OPERATION_OUTCOME,
    scope: PipelineEvidenceScope | None = None,
    operation_name: str = "conformance-op",
    recorded_at: datetime | None = None,
) -> PlatformFunctionalEvidence:
    resolved_scope = scope or sample_functional_evidence_scope()
    provenance = PipelineEvidenceProvenance(
        producer_component="diag.conformance",
        operation_id=operation_name,
        recorded_at=recorded_at or _BASE_TIME,
    )
    if kind is PipelineEvidenceKind.ARTIFACT_LINEAGE:
        payload = {
            "artifact_lineage": PipelineArtifactLineageFact(
                source_artifact_ref=ObservabilityArtifactReference(artifact_ref="doc:source"),
                derived_artifact_ref=ObservabilityArtifactReference(artifact_ref="chunk:derived"),
                lineage_operation="chunk",
            ),
        }
    elif kind is PipelineEvidenceKind.OPERATION_OUTCOME:
        payload = {
            "operation_outcome": PipelineOperationOutcomeFact(
                operation_name=operation_name,
                status=PipelineOperationStatus.SUCCEEDED,
            ),
        }
    elif kind is PipelineEvidenceKind.CANDIDATE_RANK:
        payload = {
            "candidate": PipelineCandidateFact(
                query_id="query-1",
                candidate_artifact_ref=ObservabilityArtifactReference(artifact_ref="candidate:1"),
                score=TypedPipelineScore(
                    raw_value=0.9,
                    semantics=ScoreSemantics.HIGHER_IS_BETTER,
                ),
                rank=1,
                selected=True,
            ),
        }
    elif kind is PipelineEvidenceKind.SELECTION:
        payload = {
            "selection": PipelineSelectionFact(
                query_id="query-1",
                selected_artifact_ref=ObservabilityArtifactReference(artifact_ref="selected:1"),
                candidate_count=3,
                selection_reason="top_score",
            ),
        }
    elif kind is PipelineEvidenceKind.OUTPUT_RELATION:
        payload = {
            "output_relation": PipelineOutputRelationFact(
                selected_artifact_ref=ObservabilityArtifactReference(artifact_ref="selected:1"),
                output_artifact_ref=ObservabilityArtifactReference(artifact_ref="output:1"),
                relation_kind="derived_from",
            ),
        }
    else:
        payload = {
            "validation_link": PipelineValidationLinkFact(
                validation_id=mint_event_id(),
                output_artifact_ref=ObservabilityArtifactReference(artifact_ref="output:1"),
            ),
        }
    return PlatformFunctionalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        kind=kind,
        scope=resolved_scope,
        provenance=provenance,
        **payload,
    )


def collect_all_evidence(
    store: FunctionalEvidencePersistence,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None = None,
    kind: PipelineEvidenceKind | None = None,
    page_size: int = 2,
) -> tuple[PlatformFunctionalEvidence, ...]:
    items: list[PlatformFunctionalEvidence] = []
    cursor: str | None = None
    while True:
        page = store.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                kind=kind,
                page_size=page_size,
                cursor=cursor,
            )
        )
        items.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return tuple(items)


def assert_functional_evidence_persistence_conformance(
    store: FunctionalEvidencePersistence,
    *,
    label: str,
) -> None:
    tenant_a = f"{label}-tenant-a"
    tenant_b = f"{label}-tenant-b"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    scope_a = sample_functional_evidence_scope(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a,
    )
    scope_b = sample_functional_evidence_scope(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_b,
    )
    foreign_scope = sample_functional_evidence_scope(tenant_id=tenant_b)

    first = sample_functional_evidence(
        scope=scope_a,
        operation_name="first",
        recorded_at=_BASE_TIME,
    )
    second = sample_functional_evidence(
        scope=scope_b,
        operation_name="second",
        recorded_at=_BASE_TIME + timedelta(seconds=1),
    )
    foreign = sample_functional_evidence(scope=foreign_scope, operation_name="foreign")

    store.append(first)
    store.append(second)
    store.append(foreign)
    duplicate = store.append(first)

    assert duplicate == first, f"{label}: idempotent append must return canonical record"
    collected = collect_all_evidence(
        store,
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
    )
    assert len(collected) == 2, f"{label}: expected 2 scoped records, got {len(collected)}"
    assert {item.evidence_id for item in collected} == {first.evidence_id, second.evidence_id}
    assert list(collected) == sorted(collected, key=functional_evidence_query_order_key)
    assert collect_all_evidence(
        store,
        tenant_id=tenant_b,
        task_id=foreign_scope.task_id,
        run_id=foreign_scope.run_id,
    ) == (foreign,)
    attempt_filtered = collect_all_evidence(
        store,
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a,
    )
    assert attempt_filtered == (first,)


def assert_functional_evidence_conflicting_append_fails_closed(
    store: FunctionalEvidencePersistence,
    *,
    label: str,
) -> None:
    scope = sample_functional_evidence_scope(tenant_id=f"{label}-conflict")
    evidence_id = mint_event_id()
    original = sample_functional_evidence(
        evidence_id=evidence_id,
        scope=scope,
        operation_name="original",
    )
    conflicting = sample_functional_evidence(
        evidence_id=evidence_id,
        scope=scope,
        operation_name="conflicting",
    )
    store.append(original)
    try:
        store.append(conflicting)
    except FunctionalEvidencePersistenceConflictError:
        return
    raise AssertionError(f"{label}: conflicting append must raise ConflictError")


def assert_functional_evidence_tenant_run_isolation(
    store: FunctionalEvidencePersistence,
    *,
    label: str,
) -> None:
    task_a = mint_task_id()
    run_a = mint_run_id()
    task_b = mint_task_id()
    run_b = mint_run_id()
    tenant = f"{label}-isolation"
    first = sample_functional_evidence(
        scope=sample_functional_evidence_scope(
            tenant_id=tenant,
            task_id=task_a,
            run_id=run_a,
        ),
    )
    second = sample_functional_evidence(
        scope=sample_functional_evidence_scope(
            tenant_id=tenant,
            task_id=task_b,
            run_id=run_b,
        ),
    )
    store.append(first)
    store.append(second)
    assert collect_all_evidence(
        store,
        tenant_id=tenant,
        task_id=task_a,
        run_id=run_a,
    ) == (first,)
    assert collect_all_evidence(
        store,
        tenant_id=tenant,
        task_id=task_b,
        run_id=run_b,
    ) == (second,)


def assert_functional_evidence_cross_domain_round_trip(
    store: FunctionalEvidencePersistence,
    *,
    label: str,
) -> None:
    scope = sample_functional_evidence_scope(tenant_id=f"{label}-cross-domain")
    fixtures = tuple(
        sample_functional_evidence(
            kind=kind,
            scope=scope,
            operation_name=f"{label}-{kind.value}",
            recorded_at=_BASE_TIME + timedelta(seconds=index),
        )
        for index, kind in enumerate(PipelineEvidenceKind)
    )
    for evidence in fixtures:
        store.append(evidence)
    collected = collect_all_evidence(
        store,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        page_size=1,
    )
    assert collected == fixtures


def assert_functional_evidence_corrupt_index_fails_closed(
    store: FunctionalEvidencePersistence,
    *,
    label: str,
    corrupt_index: object,
) -> None:
    scope = sample_functional_evidence_scope(tenant_id=f"{label}-corrupt")
    evidence = sample_functional_evidence(scope=scope)
    store.append(evidence)
    corrupt_index(scope=scope, evidence=evidence)
    try:
        collect_all_evidence(
            store,
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    except FunctionalEvidencePersistenceIntegrityError:
        return
    raise AssertionError(f"{label}: corrupt index must fail closed")


__all__ = [
    "assert_functional_evidence_conflicting_append_fails_closed",
    "assert_functional_evidence_corrupt_index_fails_closed",
    "assert_functional_evidence_cross_domain_round_trip",
    "assert_functional_evidence_persistence_conformance",
    "assert_functional_evidence_tenant_run_isolation",
    "collect_all_evidence",
    "sample_functional_evidence",
    "sample_functional_evidence_scope",
]
