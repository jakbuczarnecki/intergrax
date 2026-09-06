# © Artur Czarnecki. All rights reserved.

"""DG-002 R1 bounded causal evidence paging tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.observability.causal_evidence_persistence import (
    CAUSAL_EVIDENCE_QUERY_MAX_LIMIT,
    CausalEvidencePersistenceIntegrityError,
    validate_causal_evidence_query_limit,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    DocumentStoreCausalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import (
    assert_causal_evidence_paging_conformance,
    sample_causal_evidence,
)

pytestmark = pytest.mark.unit

_CURSOR_SECRET = b"causal-evidence-paging-test-secret-32b"


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        (
            "memory",
            lambda: InMemoryCausalEvidencePersistence(cursor_secret=_CURSOR_SECRET),
        ),
        (
            "document_store",
            lambda: DocumentStoreCausalEvidencePersistence(
                InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET),
                cursor_secret=_CURSOR_SECRET,
            ),
        ),
    ],
)
def test_causal_evidence_paging_conformance_matrix(label: str, factory) -> None:
    store = factory()
    try:
        assert_causal_evidence_paging_conformance(store, label=label)
    finally:
        store.close()


def test_validate_causal_evidence_query_limit_contract() -> None:
    assert validate_causal_evidence_query_limit(1) == 1
    assert validate_causal_evidence_query_limit(CAUSAL_EVIDENCE_QUERY_MAX_LIMIT) == 1000
    with pytest.raises(TypeError):
        validate_causal_evidence_query_limit(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_causal_evidence_query_limit(0)
    with pytest.raises(ValueError):
        validate_causal_evidence_query_limit(1001)


def test_document_store_first_page_query_count_bounded() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    persistence = DocumentStoreCausalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )
    tenant_id = "tenant-bounded-query"
    provider = "celery"
    transport_task_id = "bounded-transport"
    task_id = mint_task_id()
    run_id = mint_run_id()
    for _ in range(100):
        persistence.append(
            sample_causal_evidence(
                tenant_id=tenant_id,
                provider=provider,
                transport_task_id=transport_task_id,
                task_id=task_id,
                run_id=run_id,
            ),
        )

    store._rows  # touch store
    before = store.last_query_rows_examined
    query_calls = 0
    original_query = store.query

    def counting_query(*args, **kwargs):
        nonlocal query_calls
        query_calls += 1
        return original_query(*args, **kwargs)

    store.query = counting_query  # type: ignore[method-assign]
    try:
        page = persistence.page_for_transport_task(
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
            limit=10,
        )
    finally:
        store.query = original_query  # type: ignore[method-assign]

    assert len(page.items) == 10
    assert query_calls <= 3, f"expected bounded query count, got {query_calls}"
    assert store.last_query_rows_examined - before <= 15


def test_high_water_excludes_concurrent_append() -> None:
    persistence = InMemoryCausalEvidencePersistence(cursor_secret=_CURSOR_SECRET)
    tenant_id = "tenant-high-water"
    provider = "celery"
    transport_task_id = "hw-transport"
    task_id = mint_task_id()
    run_id = mint_run_id()
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        sample_causal_evidence(
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
            task_id=task_id,
            run_id=run_id,
        ).model_copy(
            update={"recorded_at": base_time.replace(second=index)},
        )
        for index in range(4)
    ]
    for record in records:
        persistence.append(record)

    page1 = persistence.page_for_transport_task(
        tenant_id=tenant_id,
        provider=provider,
        transport_task_id=transport_task_id,
        limit=2,
    )
    assert page1.next_cursor is not None
    late = sample_causal_evidence(
        tenant_id=tenant_id,
        provider=provider,
        transport_task_id=transport_task_id,
        task_id=task_id,
        run_id=run_id,
    ).model_copy(update={"recorded_at": base_time.replace(second=30)})
    persistence.append(late)

    collected = list(page1.items)
    cursor = page1.next_cursor
    while cursor is not None:
        page = persistence.page_for_transport_task(
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
            limit=2,
            cursor=cursor,
        )
        collected.extend(page.items)
        cursor = page.next_cursor

    assert late.evidence_id not in {item.evidence_id for item in collected}
    assert len(collected) == 4


def test_idempotent_append_repairs_v2_index() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    persistence = DocumentStoreCausalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )
    evidence = sample_causal_evidence(
        tenant_id="tenant-repair-v2",
        provider="celery",
        transport_task_id="repair-transport",
    )
    persistence.append(evidence)
    partition_key = f"intergrax.causal_evidence.v1:{evidence.tenant_id}"
    v2_key = persistence._transport_v2_index_document(  # noqa: SLF001
        evidence=evidence,
        partition_key=partition_key,
    ).row_key
    store.delete(partition_key, v2_key)
    persistence.append(evidence)
    page = persistence.page_for_transport_task(
        tenant_id=evidence.tenant_id,
        provider=evidence.source.provider,
        transport_task_id=evidence.source.task_id,
        limit=10,
    )
    assert page.items == (evidence,)


def test_malformed_v2_index_fails_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    persistence = DocumentStoreCausalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )
    evidence = sample_causal_evidence(
        tenant_id="tenant-malformed-v2",
        provider="celery",
        transport_task_id="malformed-v2-transport",
    )
    persistence.append(evidence)
    partition_key = f"intergrax.causal_evidence.v1:{evidence.tenant_id}"
    row_key = persistence._transport_v2_index_document(  # noqa: SLF001
        evidence=evidence,
        partition_key=partition_key,
    ).row_key
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "schema_version": "intergrax.causal_evidence.index.v2",
                "evidence_id": str(evidence.evidence_id),
                "recorded_at": datetime(1999, 1, 1, tzinfo=timezone.utc).isoformat(),
            },
        ),
    )
    with pytest.raises(CausalEvidencePersistenceIntegrityError):
        persistence.page_for_transport_task(
            tenant_id=evidence.tenant_id,
            provider=evidence.source.provider,
            transport_task_id=evidence.source.task_id,
            limit=10,
        )
