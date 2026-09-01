# © Artur Czarnecki. All rights reserved.

"""Provider qualification discovery tests (PROVIDER-QUAL-4)."""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.core.qualification import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
    QualificationEvidence,
    QualificationRunId,
    QualificationStatus,
    new_qualification_run_id,
)
from intergrax.core.qualification.discovery import (
    ProviderQualificationDiscoveryError,
    ProviderQualificationRunFilter,
    discover_provider_qualification_runs,
    run_matches_filter,
    sort_provider_qualification_runs,
)
from intergrax.core.qualification.persistence import (
    DocumentStoreProviderQualificationPersistence,
    ProviderQualificationPersistenceIntegrityError,
    provider_qualification_run_to_proof_receipt,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.proofs.receipts.document_store import (
    proof_receipt_lookup_row_key,
    proof_receipt_partition_key,
    proof_receipt_to_document,
)

pytestmark = pytest.mark.unit

_BASE_TIME = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def _subject(
    *,
    provider_id: str = "postgresql",
    provider_version: str = "16.6",
    capability_id: str = "collaborative_work.persistence.v1",
    domain: str = "collaborative_work",
    qualification_suite_id: str = "cw.postgresql.repository.v1",
    qualification_suite_version: str = "1.0.0",
    environment_id: str = "local-docker-qual-host",
) -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id=provider_id,
        provider_version=provider_version,
        capability_id=capability_id,
        domain=domain,
        intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        qualification_suite_id=qualification_suite_id,
        qualification_suite_version=qualification_suite_version,
        environment_id=environment_id,
        adapter_identity=f"intergrax.integrations.providers.{provider_id}",
    )


def _run(
    *,
    run_id: QualificationRunId | None = None,
    provider_id: str = "postgresql",
    provider_version: str = "16.6",
    capability_id: str = "collaborative_work.persistence.v1",
    qualification_suite_id: str = "cw.postgresql.repository.v1",
    status: QualificationStatus = QualificationStatus.PRODUCTION_QUALIFIED,
    executed_at: datetime = _BASE_TIME,
) -> ProviderQualificationRun:
    return ProviderQualificationRun(
        qualification_run_id=run_id or new_qualification_run_id(),
        subject=_subject(
            provider_id=provider_id,
            provider_version=provider_version,
            capability_id=capability_id,
            qualification_suite_id=qualification_suite_id,
        ),
        status=status,
        executed_at=executed_at,
        executor=ProviderQualificationExecutor(
            executor_kind="local_cli",
            executor_id="qual-host-01",
            executor_version="2026.08.17",
        ),
        result_summary=ProviderQualificationResultSummary(
            passed=42,
            failed=0,
            skipped=3,
            label=qualification_suite_id,
        ),
        evidence=(
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="suite.passed",
                ref="tests/integration/cw/test_repository.py",
            ),
        ),
        reproducibility="uv run pytest tests/integration/cw/test_repository.py",
        limitations=("bounded local docker host",),
        source_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
            bounded_environment="docker-host",
        ),
    )


def _persist_many(
    persistence: DocumentStoreProviderQualificationPersistence,
    *runs: ProviderQualificationRun,
) -> None:
    for item in runs:
        persistence.persist(item)


def test_persist_multiple_runs_and_query_by_provider_id() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    pg_run = _run(provider_id="postgresql")
    sqlite_run = _run(provider_id="sqlite")
    _persist_many(persistence, pg_run, sqlite_run)

    page = persistence.find_runs(ProviderQualificationRunFilter(provider_id="postgresql"))

    assert [item.qualification_run_id for item in page.runs] == [pg_run.qualification_run_id]
    assert isinstance(page.runs[0], ProviderQualificationRun)


def test_query_by_provider_id_and_provider_version() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    match = _run(provider_id="postgresql", provider_version="16.6")
    other_version = _run(provider_id="postgresql", provider_version="15.2")
    _persist_many(persistence, match, other_version)

    page = persistence.find_runs(
        ProviderQualificationRunFilter(
            provider_id="postgresql",
            provider_version="16.6",
        ),
    )

    assert [item.qualification_run_id for item in page.runs] == [match.qualification_run_id]


def test_query_by_capability_id() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    cw_run = _run(capability_id="collaborative_work.persistence.v1")
    other_capability = _run(capability_id="rag.vector_store.v1", provider_id="sqlite")
    _persist_many(persistence, cw_run, other_capability)

    page = persistence.find_runs(
        ProviderQualificationRunFilter(capability_id="collaborative_work.persistence.v1"),
    )

    assert {item.qualification_run_id for item in page.runs} == {cw_run.qualification_run_id}


def test_query_by_qualification_suite_id() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    suite_run = _run(qualification_suite_id="cw.postgresql.repository.v1")
    other_suite = _run(
        provider_id="sqlite",
        qualification_suite_id="cw.sqlite.repository.v1",
    )
    _persist_many(persistence, suite_run, other_suite)

    page = persistence.find_runs(
        ProviderQualificationRunFilter(qualification_suite_id="cw.postgresql.repository.v1"),
    )

    assert [item.qualification_run_id for item in page.runs] == [suite_run.qualification_run_id]


def test_combined_filters_return_only_correct_runs() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    target = _run(
        provider_id="postgresql",
        provider_version="16.6",
        capability_id="collaborative_work.persistence.v1",
        qualification_suite_id="cw.postgresql.repository.v1",
    )
    partial_match = _run(
        provider_id="postgresql",
        provider_version="15.2",
        capability_id="collaborative_work.persistence.v1",
        qualification_suite_id="cw.postgresql.repository.v1",
    )
    other_provider = _run(provider_id="sqlite")
    _persist_many(persistence, target, partial_match, other_provider)

    page = persistence.find_runs(
        ProviderQualificationRunFilter(
            provider_id="postgresql",
            provider_version="16.6",
            capability_id="collaborative_work.persistence.v1",
            qualification_suite_id="cw.postgresql.repository.v1",
        ),
    )

    assert [item.qualification_run_id for item in page.runs] == [target.qualification_run_id]


def test_unknown_provider_returns_empty_result() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    _persist_many(persistence, _run(provider_id="postgresql"))

    page = persistence.find_runs(ProviderQualificationRunFilter(provider_id="oracle"))

    assert page.runs == ()
    assert page.next_cursor is None


def test_results_reconstruct_as_provider_qualification_run() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    run = _run()
    persistence.persist(run)

    page = persistence.find_runs(ProviderQualificationRunFilter(provider_id="postgresql"))

    assert page.runs == (run,)


def test_deterministic_ordering_by_executed_at_then_run_id() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    older = _run(
        run_id=QualificationRunId("qual_run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        executed_at=_BASE_TIME,
    )
    newer = _run(
        run_id=QualificationRunId("qual_run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        executed_at=_BASE_TIME + timedelta(hours=1),
    )
    same_time_high_id = _run(
        run_id=QualificationRunId("qual_run_ccccccccccccccccccccccccccccccc0"),
        executed_at=_BASE_TIME + timedelta(hours=1),
    )
    _persist_many(persistence, older, newer, same_time_high_id)

    page = persistence.find_runs(ProviderQualificationRunFilter(provider_id="postgresql"))

    assert [item.qualification_run_id for item in page.runs] == [
        same_time_high_id.qualification_run_id,
        newer.qualification_run_id,
        older.qualification_run_id,
    ]
    assert sort_provider_qualification_runs(page.runs) == page.runs


def test_provider_neutrality_with_two_provider_ids() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    postgresql = _run(provider_id="postgresql")
    sqlite = _run(provider_id="sqlite")
    _persist_many(persistence, postgresql, sqlite)

    pg_page = persistence.find_runs(ProviderQualificationRunFilter(provider_id="postgresql"))
    sqlite_page = persistence.find_runs(ProviderQualificationRunFilter(provider_id="sqlite"))

    assert [item.qualification_run_id for item in pg_page.runs] == [
        postgresql.qualification_run_id,
    ]
    assert [item.qualification_run_id for item in sqlite_page.runs] == [
        sqlite.qualification_run_id,
    ]


def test_no_vendor_specific_dispatch_in_discovery_core() -> None:
    import intergrax.core.qualification.discovery as discovery_module

    source = inspect.getsource(discovery_module)
    for forbidden in (
        "find_postgresql",
        "find_oracle",
        "find_sqlite",
        "postgresql",
        "oracle",
    ):
        assert forbidden not in source


def test_pagination_returns_bounded_pages_with_cursor() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    runs = tuple(
        _run(
            run_id=QualificationRunId(f"qual_run_{index:032x}"),
            executed_at=_BASE_TIME + timedelta(minutes=index),
        )
        for index in range(5)
    )
    _persist_many(persistence, *runs)

    first = persistence.find_runs(
        ProviderQualificationRunFilter(provider_id="postgresql"),
        limit=2,
    )
    second = persistence.find_runs(
        ProviderQualificationRunFilter(provider_id="postgresql"),
        limit=2,
        cursor=first.next_cursor,
    )
    third = persistence.find_runs(
        ProviderQualificationRunFilter(provider_id="postgresql"),
        limit=2,
        cursor=second.next_cursor,
    )

    assert len(first.runs) == 2
    assert len(second.runs) == 2
    assert len(third.runs) == 1
    assert first.next_cursor is not None
    assert second.next_cursor is not None
    assert third.next_cursor is None
    assert [item.qualification_run_id for item in first.runs + second.runs + third.runs] == [
        item.qualification_run_id for item in reversed(runs)
    ]


def test_corrupt_qualification_record_fails_closed_during_discovery() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    run = _run()
    persistence.persist(run)
    corrupt = proof_receipt_to_document(
        provider_qualification_run_to_proof_receipt(run).model_copy(
            update={"domain_evidence": {"schema_version": "broken"}},
        ),
    )
    store.put(corrupt)

    with pytest.raises(ProviderQualificationPersistenceIntegrityError):
        persistence.find_runs(ProviderQualificationRunFilter(provider_id="postgresql"))


def test_discovery_requires_at_least_one_filter_criterion() -> None:
    store = InMemoryDocumentStore()

    with pytest.raises(
        ProviderQualificationDiscoveryError,
        match="at least one filter criterion",
    ):
        discover_provider_qualification_runs(store, ProviderQualificationRunFilter())


def test_exact_match_does_not_use_substring_guessing() -> None:
    run = _run(provider_id="postgresql")
    assert run_matches_filter(run, ProviderQualificationRunFilter(provider_id="postgresql"))
    assert not run_matches_filter(run, ProviderQualificationRunFilter(provider_id="post"))


def test_platform_reuse_assertion_no_parallel_qualification_index() -> None:
    import intergrax.core.qualification.discovery as discovery_module

    source = inspect.getsource(discovery_module)
    assert "ProofReceipt" in source or "proof_receipt" in source
    assert "DocumentStore" in source or "document_store" in source
    assert "ProviderQualificationIndex" not in source
    assert "ProviderQualificationDatabase" not in source
    assert "QualificationSearchDatabase" not in source


def test_malformed_proof_receipt_schema_fails_closed_during_discovery() -> None:
    store = InMemoryDocumentStore()
    partition_key = proof_receipt_partition_key("intergrax.provider_qualification")
    row_key = proof_receipt_lookup_row_key(
        "provider_qualification",
        str(new_qualification_run_id()),
    )
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"schema_version": "intergrax.proof_receipt.v9"},
        ),
    )

    with pytest.raises(ProviderQualificationPersistenceIntegrityError):
        discover_provider_qualification_runs(
            store,
            ProviderQualificationRunFilter(provider_id="postgresql"),
        )


def test_optional_status_filter() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    qualified = _run(status=QualificationStatus.QUALIFIED)
    rejected = replace(_run(), status=QualificationStatus.REJECTED)
    _persist_many(persistence, qualified, rejected)

    page = persistence.find_runs(
        ProviderQualificationRunFilter(
            provider_id="postgresql",
            status=QualificationStatus.REJECTED,
        ),
    )

    assert [item.qualification_run_id for item in page.runs] == [
        rejected.qualification_run_id,
    ]
