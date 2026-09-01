# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-1-R2 closeout: honest InMemory qualification and boundedness proofs."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.functional_evidence_bounds import MAX_SUPPORTING_EVIDENCE_REFS
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidenceQueryRequest,
)
from intergrax.runtime.diagnostics.functional_evidence_reconstruction import (
    FunctionalEvidenceReconstructor,
)
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)

pytestmark = pytest.mark.unit

_TEST_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def _persistence() -> InMemoryFunctionalEvidencePersistence:
    return InMemoryFunctionalEvidencePersistence(cursor_secret=_TEST_CURSOR_SECRET)


def _scope() -> PipelineEvidenceScope:
    return PipelineEvidenceScope(
        tenant_id="tenant-r2",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )


def _operation_evidence(
    scope: PipelineEvidenceScope,
    *,
    index: int,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.generic",
            operation_id=f"op-{index}",
            recorded_at=_BASE_TIME + timedelta(seconds=index),
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name=f"op-{index}",
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )


def test_inmemory_append_uses_bisect_insort_on_list() -> None:
    source = inspect.getsource(InMemoryFunctionalEvidencePersistence.append)
    assert "bisect.insort" in source
    assert "list" in inspect.getsource(InMemoryFunctionalEvidencePersistence)


def test_inmemory_is_not_scale_qualified_provider() -> None:
    doc = InMemoryFunctionalEvidencePersistence.__doc__ or ""
    module_doc = inspect.getmodule(InMemoryFunctionalEvidencePersistence).__doc__ or ""
    combined = f"{module_doc}\n{doc}".lower()
    assert "not" in combined and "scale" in combined
    assert "conformance" in combined or "unit" in combined


def test_query_page_is_bounded_and_keyset_correct() -> None:
    persistence = _persistence()
    scope = _scope()
    for index in range(25):
        persistence.append(_operation_evidence(scope, index=index))

    page_size = 7
    first = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=page_size,
        ),
    )
    assert len(first.items) == page_size
    assert first.next_cursor is not None

    collected: list[str] = [item.evidence_id for item in first.items]
    cursor = first.next_cursor
    while cursor is not None:
        page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=page_size,
                cursor=cursor,
            ),
        )
        collected.extend(item.evidence_id for item in page.items)
        cursor = page.next_cursor

    assert len(collected) == 25
    assert len(set(collected)) == 25


def test_reconstruction_retains_bounded_supporting_refs() -> None:
    persistence = _persistence()
    scope = _scope()
    for index in range(MAX_SUPPORTING_EVIDENCE_REFS + 12):
        persistence.append(_operation_evidence(scope, index=index))

    reconstruction = FunctionalEvidenceReconstructor(persistence).reconstruct(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        required_kinds=frozenset({PipelineEvidenceKind.OPERATION_OUTCOME}),
    )

    assert reconstruction.evidence_summary.total_evidence_count == MAX_SUPPORTING_EVIDENCE_REFS + 12
    assert len(reconstruction.supporting_evidence_refs) <= MAX_SUPPORTING_EVIDENCE_REFS


def test_filtered_query_scans_without_resorting_history() -> None:
    persistence = _persistence()
    scope = _scope()
    attempt_scope = PipelineEvidenceScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=mint_attempt_id(),
    )
    for index in range(20):
        persistence.append(_operation_evidence(scope, index=index))
    persistence.append(_operation_evidence(attempt_scope, index=99))

    page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=attempt_scope.attempt_id,
            page_size=10,
        ),
    )
    assert len(page.items) == 1
    assert page.next_cursor is None
