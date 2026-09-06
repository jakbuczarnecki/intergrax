# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticScopeDiscoveryIntegrityError,
    DiagnosticScopeDiscoveryRequest,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    ProblemScopeReference,
    TransportScopeReference,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeProviderIntegrityError,
    assert_diagnostic_scope_discovery_provider_conformance,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_service import (
    DiagnosticScopeDiscoveryService,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_grouping import problem_grouping_subject_ref_for_execution
from intergrax.runtime.diagnostics.providers.causal_transport_scope_provider import (
    CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
    CausalTransportScopeProvider,
)
from intergrax.runtime.diagnostics.providers.problem_scope_provider import ProblemScopeProvider
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePage,
    CausalEvidencePersistence,
    CausalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    DocumentStoreCausalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    create_problem_for_tests,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_PROVIDER = "celery"
_TRANSPORT_TASK_ID = "celery-task-1"
_RECORDED_AT = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)


def _recorded_at_for_index(index: int) -> datetime:
    return _RECORDED_AT + timedelta(microseconds=index)


def _transport_ref(
    *,
    tenant_id: str = _TENANT,
    provider: str = _PROVIDER,
    transport_task_id: str = _TRANSPORT_TASK_ID,
) -> MessageBusTaskRef:
    return MessageBusTaskRef(
        provider=provider,
        task_id=transport_task_id,
        tenant_id=tenant_id,
    )


def _execution_ref(
    *,
    tenant_id: str = _TENANT,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
) -> RuntimeExecutionRef:
    return RuntimeExecutionRef(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )


def _causal_evidence(
    *,
    tenant_id: str = _TENANT,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    provider: str = _PROVIDER,
    transport_task_id: str = _TRANSPORT_TASK_ID,
    relation_kind: CausalRelationKind = CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
    recorded_at: datetime = _RECORDED_AT,
    evidence_id: str | None = None,
) -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        relation_kind=relation_kind,
        tenant_id=tenant_id,
        source=_transport_ref(
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
        ),
        target=_execution_ref(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        recorded_at=recorded_at,
    )


def _provider(
    persistence: CausalEvidencePersistence | None = None,
) -> CausalTransportScopeProvider:
    return CausalTransportScopeProvider(
        causal_evidence_persistence=persistence or InMemoryCausalEvidencePersistence(),
    )


def _service(
    persistence: CausalEvidencePersistence | None = None,
) -> DiagnosticScopeDiscoveryService:
    return DiagnosticScopeDiscoveryService(
        providers=(_provider(persistence),),
    )


def _transport_request(
    *,
    transport_task_id: str = _TRANSPORT_TASK_ID,
    candidate_limit: int = 10,
) -> DiagnosticScopeDiscoveryRequest:
    return DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=TransportScopeReference(
            provider=_PROVIDER,
            transport_task_id=transport_task_id,
        ),
        candidate_limit=candidate_limit,
    )


def _seed_evidence(
    persistence: InMemoryCausalEvidencePersistence,
    *evidence_records: PlatformCausalEvidence,
) -> InMemoryCausalEvidencePersistence:
    for evidence in evidence_records:
        persistence.append(evidence)
    return persistence


def _single_page_persistence(
    items: tuple[PlatformCausalEvidence, ...],
    *,
    next_cursor: str | None = None,
) -> MagicMock:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.return_value = CausalEvidencePage(
        items=items,
        next_cursor=next_cursor,
    )
    return persistence


def _sequential_page_persistence(
    *pages: CausalEvidencePage,
) -> MagicMock:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = list(pages)
    return persistence


@dataclass
class _OffsetPagedTransportPersistence:
    records: tuple[PlatformCausalEvidence, ...]
    trailing_cursor: str | None = None
    page_calls: list[dict[str, object]] = field(default_factory=list)
    list_calls: int = 0

    def page_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:
        self.page_calls.append(
            {
                "tenant_id": tenant_id,
                "provider": provider,
                "transport_task_id": transport_task_id,
                "limit": limit,
                "cursor": cursor,
            },
        )
        start = 0 if cursor is None else int(cursor)
        end = min(start + limit, len(self.records))
        items = self.records[start:end]
        if end < len(self.records):
            next_cursor = str(end)
        else:
            next_cursor = self.trailing_cursor
        return CausalEvidencePage(items=tuple(items), next_cursor=next_cursor)

    def list_for_transport_task(self, **kwargs: object) -> tuple[PlatformCausalEvidence, ...]:
        self.list_calls += 1
        return self.records


def test_no_evidence_returns_not_found() -> None:
    persistence = InMemoryCausalEvidencePersistence()
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.NOT_FOUND
    assert result.candidate_count == 0
    assert result.candidate_count_exact is True


def test_one_evidence_returns_resolved() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=mint_attempt_id()),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.resolved_scope is not None
    assert result.resolved_scope.task_id == task_id
    assert result.resolved_scope.run_id == run_id
    assert result.candidate_count == 1


def test_duplicate_evidence_same_task_run_returns_resolved() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    persistence = InMemoryCausalEvidencePersistence()
    _seed_evidence(
        persistence,
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
        ),
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=datetime(2026, 6, 8, 12, 1, 0, tzinfo=UTC),
        ),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1


def test_multiple_attempts_same_task_run_returns_resolved() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
        ),
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            recorded_at=datetime(2026, 6, 8, 12, 1, 0, tzinfo=UTC),
        ),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1


def test_two_runs_returns_ambiguous() -> None:
    task_id = mint_task_id()
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(task_id=task_id, run_id=mint_run_id(), attempt_id=mint_attempt_id()),
        _causal_evidence(task_id=task_id, run_id=mint_run_id(), attempt_id=mint_attempt_id()),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 2
    assert result.candidate_count_exact is True


def test_two_tasks_returns_ambiguous() -> None:
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
        _causal_evidence(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 2


def test_candidate_limit_preserves_truth_count() -> None:
    persistence = InMemoryCausalEvidencePersistence()
    for _ in range(5):
        persistence.append(
            _causal_evidence(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
            ),
        )
    result = _service(persistence).discover_scope(
        _transport_request(candidate_limit=2),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 5
    assert result.candidate_count_exact is True
    assert len(result.candidates) == 2


def test_tenant_mismatch_raises_integrity_error() -> None:
    evidence = _causal_evidence(
        tenant_id="tenant-b",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    persistence = _single_page_persistence((evidence,))
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="tenant"):
        _service(persistence).discover_scope(_transport_request())
    persistence.list_for_transport_task.assert_not_called()


def test_source_provider_mismatch_raises_integrity_error() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        provider="rabbitmq",
    )
    persistence = _single_page_persistence((evidence,))
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="provider"):
        _service(persistence).discover_scope(_transport_request())


def test_source_transport_id_mismatch_raises_integrity_error() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        transport_task_id="other-task",
    )
    persistence = _single_page_persistence((evidence,))
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="transport_task_id"):
        _service(persistence).discover_scope(_transport_request())


def test_wrong_relation_kind_raises_integrity_error() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    faulty = PlatformCausalEvidence.model_construct(
        schema_version=evidence.schema_version,
        evidence_id=evidence.evidence_id,
        relation_kind="wrong.relation",
        tenant_id=evidence.tenant_id,
        source=evidence.source,
        target=evidence.target,
        recorded_at=evidence.recorded_at,
    )
    persistence = _single_page_persistence((faulty,))
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="relation_kind"):
        _service(persistence).discover_scope(_transport_request())


def test_persistence_integrity_maps_to_provider_integrity() -> None:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = CausalEvidencePersistenceIntegrityError(
        "bad causal index",
    )
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="bad causal index"):
        _service(persistence).discover_scope(_transport_request())


def test_connection_error_returns_provider_unavailable() -> None:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = ConnectionError("store down")
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_timeout_error_returns_provider_unavailable() -> None:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = TimeoutError("store timeout")
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_unexpected_value_error_propagates() -> None:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = ValueError("programming bug")
    with pytest.raises(ValueError, match="programming bug"):
        _service(persistence).discover_scope(_transport_request())


def test_provider_conformance() -> None:
    assert_diagnostic_scope_discovery_provider_conformance(
        _provider(),
        expected_provider_id=CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
        expected_reference_kind=DiagnosticScopeReferenceKind.TRANSPORT,
    )


def test_provenance_uses_first_evidence_in_query_order_for_duplicate_scope() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    first_id = mint_event_id()
    second_id = mint_event_id()
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=first_id,
            recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
        ),
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=second_id,
            recorded_at=datetime(2026, 6, 8, 12, 1, 0, tzinfo=UTC),
        ),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidates[0].provenance.canonical_record_ref == f"causal_evidence:{first_id}"


def test_deterministic_candidate_order() -> None:
    task_a = mint_task_id()
    task_b = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(task_id=task_b, run_id=run_b, attempt_id=mint_attempt_id()),
        _causal_evidence(task_id=task_a, run_id=run_a, attempt_id=mint_attempt_id()),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    ordered = [(str(c.subject_ref.task_id), str(c.subject_ref.run_id)) for c in result.candidates]
    assert ordered == sorted(ordered)


def test_pluginability_problem_and_transport_providers_share_unchanged_service() -> None:
    store = in_memory_document_store_for_problem_tests()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    causal_persistence = InMemoryCausalEvidencePersistence()

    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem = create_problem_for_tests(
        problem_persistence,
        sample_problem(tenant_id=_TENANT, subject_refs=(subject,), occurrence_count=0),
        indexed_subject_refs=(subject,),
    )

    task_id = mint_task_id()
    run_id = mint_run_id()
    causal_persistence.append(
        _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=mint_attempt_id()),
    )

    service = DiagnosticScopeDiscoveryService(
        providers=(
            ProblemScopeProvider(
                problem_persistence=problem_persistence,
                occurrence_persistence=occurrence_persistence,
            ),
            CausalTransportScopeProvider(causal_evidence_persistence=causal_persistence),
        ),
    )

    problem_result = service.discover_scope(
        DiagnosticScopeDiscoveryRequest(
            tenant_id=_TENANT,
            reference=ProblemScopeReference(problem_id=problem.problem_id),
        ),
    )
    assert problem_result.status is DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE

    transport_result = service.discover_scope(_transport_request())
    assert transport_result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert transport_result.resolved_scope is not None
    assert transport_result.resolved_scope.task_id == task_id


def test_malformed_target_raises_integrity_error() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    faulty_target = RuntimeExecutionRef.model_construct(
        task_id="not-a-valid-task-id",
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        tenant_id=_TENANT,
    )
    faulty = PlatformCausalEvidence.model_construct(
        schema_version=evidence.schema_version,
        evidence_id=evidence.evidence_id,
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=evidence.tenant_id,
        source=evidence.source,
        target=faulty_target,
        recorded_at=evidence.recorded_at,
    )
    persistence = _single_page_persistence((faulty,))
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="execution diagnostic scope"):
        _service(persistence).discover_scope(_transport_request())


def test_provider_source_uses_page_api_only() -> None:
    source = Path(
        "intergrax/runtime/diagnostics/providers/causal_transport_scope_provider.py",
    ).read_text(encoding="utf-8")
    assert "page_for_transport_task(" in source
    assert "list_for_transport_task(" not in source


def test_page_api_only_structural_guard() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    fake = _OffsetPagedTransportPersistence(records=(evidence,))
    _service(fake).discover_scope(_transport_request())
    assert fake.list_calls == 0
    assert len(fake.page_calls) == 1


def test_hundred_one_evidence_same_scope_uses_two_pages() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    records = tuple(
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=_recorded_at_for_index(index),
        )
        for index in range(101)
    )
    fake = _OffsetPagedTransportPersistence(records=records)
    result = _service(fake).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1
    assert result.candidate_count_exact is True
    assert len(fake.page_calls) == 2


def test_second_page_introduces_ambiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.causal_transport_scope_provider._CAUSAL_EVIDENCE_PAGE_SIZE",
        1,
    )
    first_scope = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )
    second_scope = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 1, tzinfo=UTC),
    )
    fake = _OffsetPagedTransportPersistence(records=(first_scope, second_scope))
    result = _service(fake).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 2
    assert len(fake.page_calls) == 2


def test_duplicate_scope_across_pages_dedupes() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    first = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )
    duplicate = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 1, tzinfo=UTC),
    )
    records = tuple([first] + [duplicate] * 100)
    fake = _OffsetPagedTransportPersistence(records=records)
    result = _service(fake).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1
    assert len(fake.page_calls) == 2


def test_earliest_provenance_survives_duplicate_on_later_page() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    first_id = mint_event_id()
    first = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        evidence_id=first_id,
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )
    duplicate = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 1, tzinfo=UTC),
    )
    records = tuple([first] + [duplicate] * 100)
    fake = _OffsetPagedTransportPersistence(records=records)
    result = _service(fake).discover_scope(_transport_request())
    assert result.candidates[0].provenance.canonical_record_ref == f"causal_evidence:{first_id}"


def test_exactly_thousand_evidence_complete_scan() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    records = tuple(
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=_recorded_at_for_index(index),
        )
        for index in range(1000)
    )
    fake = _OffsetPagedTransportPersistence(records=records)
    result = _service(fake).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count_exact is True
    assert len(fake.page_calls) == 10


def test_truncated_one_scope_is_insufficient_inexact() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    records = tuple(
        _causal_evidence(
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=_recorded_at_for_index(index),
        )
        for index in range(1000)
    )
    fake = _OffsetPagedTransportPersistence(
        records=records,
        trailing_cursor="more-evidence",
    )
    result = _service(fake).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE
    assert result.candidate_count == 1
    assert result.candidate_count_exact is False
    assert result.limitations


def test_truncated_two_scopes_is_ambiguous_inexact() -> None:
    scope_a = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )
    scope_b = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        evidence_id=mint_event_id(),
        recorded_at=datetime(2026, 6, 8, 12, 0, 1, tzinfo=UTC),
    )
    records = tuple([scope_a] * 500 + [scope_b] * 500)
    fake = _OffsetPagedTransportPersistence(
        records=records,
        trailing_cursor="more-evidence",
    )
    result = _service(fake).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 2
    assert result.candidate_count_exact is False


def test_budget_caps_page_calls_at_ten() -> None:
    records = tuple(
        _causal_evidence(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=_recorded_at_for_index(index),
        )
        for index in range(12000)
    )
    fake = _OffsetPagedTransportPersistence(
        records=records,
        trailing_cursor="more-evidence",
    )
    _service(fake).discover_scope(_transport_request())
    assert len(fake.page_calls) == 10


def test_remaining_budget_uses_reduced_page_limit() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    limits_seen: list[int] = []

    def page_side_effect(
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:
        limits_seen.append(limit)
        if len(limits_seen) <= 9:
            return CausalEvidencePage(
                items=tuple([evidence] * 100),
                next_cursor=f"page-{len(limits_seen)}",
            )
        if len(limits_seen) == 10:
            return CausalEvidencePage(
                items=tuple([evidence] * 50),
                next_cursor="page-10",
            )
        return CausalEvidencePage(
            items=tuple([evidence] * 50),
            next_cursor="more-evidence",
        )

    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = page_side_effect
    _service(persistence).discover_scope(_transport_request())
    assert limits_seen[10] == 50


def test_read_count_bound_with_large_available_history() -> None:
    records = tuple(
        _causal_evidence(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            evidence_id=mint_event_id(),
            recorded_at=_recorded_at_for_index(index),
        )
        for index in range(15000)
    )
    fake = _OffsetPagedTransportPersistence(
        records=records,
        trailing_cursor="more-evidence",
    )
    _service(fake).discover_scope(_transport_request())
    requested_limits = sum(call["limit"] for call in fake.page_calls)
    assert requested_limits == 1000
    assert len(fake.page_calls) == 10


def test_integrity_on_second_page_fails_whole_discover() -> None:
    valid = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    persistence = _sequential_page_persistence(
        CausalEvidencePage(items=(valid,), next_cursor="page-2"),
        CausalEvidencePersistenceIntegrityError("bad page two"),
    )
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="bad page two"):
        _service(persistence).discover_scope(_transport_request())


def test_availability_on_second_page_fails_whole_discover() -> None:
    valid = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    persistence = _sequential_page_persistence(
        CausalEvidencePage(items=(valid,), next_cursor="page-2"),
        TimeoutError("store timeout"),
    )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_os_error_maps_to_provider_unavailable() -> None:
    persistence = MagicMock(spec=CausalEvidencePersistence)
    persistence.page_for_transport_task.side_effect = OSError("disk unavailable")
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_empty_continuation_page_raises_integrity_error() -> None:
    persistence = _single_page_persistence((), next_cursor="stuck")
    with pytest.raises(
        DiagnosticScopeDiscoveryIntegrityError,
        match="empty items with continuation cursor",
    ):
        _service(persistence).discover_scope(_transport_request())


def test_repeated_cursor_raises_integrity_error() -> None:
    evidence = _causal_evidence(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    persistence = _sequential_page_persistence(
        CausalEvidencePage(items=(evidence,), next_cursor="same"),
        CausalEvidencePage(items=(evidence,), next_cursor="same"),
    )
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="cursor"):
        _service(persistence).discover_scope(_transport_request())


def test_service_is_deterministic_for_paged_transport() -> None:
    persistence = _seed_evidence(
        InMemoryCausalEvidencePersistence(),
        _causal_evidence(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
        _causal_evidence(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
    )
    service = _service(persistence)
    request = _transport_request()
    first = service.discover_scope(request)
    second = service.discover_scope(request)
    assert first == second


def test_in_memory_paged_integration_crosses_page_boundary() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    persistence = InMemoryCausalEvidencePersistence()
    for index in range(101):
        persistence.append(
            _causal_evidence(
                task_id=task_id,
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                evidence_id=mint_event_id(),
                recorded_at=_recorded_at_for_index(index),
            ),
        )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1
    assert result.candidate_count_exact is True


def test_document_store_paged_integration_crosses_page_boundary() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    persistence = DocumentStoreCausalEvidencePersistence(
        InMemoryDocumentStore(cursor_secret=b"causal-transport-scope-provider-test-32b"),
        cursor_secret=b"causal-transport-scope-provider-test-32b",
    )
    for index in range(101):
        persistence.append(
            _causal_evidence(
                task_id=task_id,
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                evidence_id=mint_event_id(),
                recorded_at=_recorded_at_for_index(index),
            ),
        )
    result = _service(persistence).discover_scope(_transport_request())
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1
    assert result.candidate_count_exact is True
