# © Artur Czarnecki. All rights reserved.

"""Focused Mongo read-path qualification for DIAG-FUNCTIONAL-READ-R1."""

from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import FunctionalEvidenceQueryRequest
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from tests.system.functional_diagnostics_scale.mongodb_backend import resolve_mongodb_uri

_CURSOR_SECRET = b"diag-functional-read-r1-qualification-secret-32b"
_EVIDENCE_COUNT = 5000
_PAGE_SIZE = 25
_BASE_TIME = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ReadR1LatencySample:
    label: str
    latency_ms: float
    item_count: int


@dataclass(frozen=True, slots=True)
class ReadR1MongoQualificationResult:
    evidence_count: int
    page_size: int
    first_page_canonical_gets: int
    first_page_index_queries: int
    latencies_ms: tuple[ReadR1LatencySample, ...]
    union_count: int
    analyzer_fidelity_percent: float
    passed: bool


class _CountingMongoStore:
    def __init__(self, inner: ConditionalDocumentStore) -> None:
        self._inner = inner
        self.get_calls = 0
        self.query_calls = 0

    @property
    def query_cursor_codec(self):
        return self._inner.query_cursor_codec

    def get(self, partition_key: str, row_key: str):
        self.get_calls += 1
        return self._inner.get(partition_key, row_key)

    def put(self, document):
        return self._inner.put(document)

    def delete(self, partition_key: str, row_key: str):
        return self._inner.delete(partition_key, row_key)

    def query(self, partition_key: str, *, limit: int, row_key_prefix: str | None = None, cursor: str | None = None):
        self.query_calls += 1
        return self._inner.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def put_if_absent(self, document):
        return self._inner.put_if_absent(document)

    def replace_if_match(self, *, expected, replacement):
        return self._inner.replace_if_match(expected=expected, replacement=replacement)

    def delete_if_match(self, *, expected):
        return self._inner.delete_if_match(expected=expected)

    def close(self):
        return self._inner.close()


def run_read_r1_mongo_qualification(
    *,
    artifact_dir: Path,
) -> ReadR1MongoQualificationResult:
    uri = resolve_mongodb_uri()
    collection = f"diag_functional_read_r1_{uuid.uuid4().hex[:12]}"
    inner = assert_conditional_document_store(
        create_mongodb_document_store(
            uri=uri,
            database="intergrax_diag_read_r1",
            collection_name=collection,
        ),
    )
    store = _CountingMongoStore(inner)
    persistence = DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id=f"read-r1-{uuid.uuid4().hex[:8]}")
    for index in range(_EVIDENCE_COUNT):
        persistence.append(
            sample_functional_evidence(
                scope=scope,
                operation_name=f"op-{index}",
                recorded_at=_BASE_TIME + timedelta(seconds=index),
            ),
        )

    latencies: list[ReadR1LatencySample] = []

    store.get_calls = 0
    store.query_calls = 0
    start = time.perf_counter()
    first_page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=_PAGE_SIZE,
        ),
    )
    latencies.append(
        ReadR1LatencySample(
            label="first_page",
            latency_ms=(time.perf_counter() - start) * 1000,
            item_count=len(first_page.items),
        ),
    )
    first_gets = store.get_calls
    first_queries = store.query_calls

    cursor = first_page.next_cursor
    middle_cursor = cursor
    for _ in range((_EVIDENCE_COUNT // _PAGE_SIZE) // 2):
        if middle_cursor is None:
            break
        middle_page_step = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=_PAGE_SIZE,
                cursor=middle_cursor,
            ),
        )
        middle_cursor = middle_page_step.next_cursor
    start = time.perf_counter()
    middle_page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=_PAGE_SIZE,
            cursor=middle_cursor,
        ),
    )
    latencies.append(
        ReadR1LatencySample(
            label="middle_page",
            latency_ms=(time.perf_counter() - start) * 1000,
            item_count=len(middle_page.items),
        ),
    )
    cursor = middle_page.next_cursor
    final_cursor: str | None = None
    while cursor is not None:
        final_cursor = cursor
        page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=_PAGE_SIZE,
                cursor=cursor,
            ),
        )
        cursor = page.next_cursor
    start = time.perf_counter()
    if final_cursor is not None:
        final_page = persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=_PAGE_SIZE,
                cursor=final_cursor,
            ),
        )
    else:
        final_page = middle_page
    latencies.append(
        ReadR1LatencySample(
            label="final_page",
            latency_ms=(time.perf_counter() - start) * 1000,
            item_count=len(final_page.items),
        ),
    )

    start = time.perf_counter()
    filtered_page = persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            kind=sample_functional_evidence(scope=scope).kind,
            page_size=_PAGE_SIZE,
        ),
    )
    latencies.append(
        ReadR1LatencySample(
            label="filtered_page",
            latency_ms=(time.perf_counter() - start) * 1000,
            item_count=len(filtered_page.items),
        ),
    )

    union = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        page_size=_PAGE_SIZE,
    )
    passed = (
        len(first_page.items) == _PAGE_SIZE
        and first_gets <= 50
        and first_gets < _EVIDENCE_COUNT
        and len(union) == _EVIDENCE_COUNT
    )
    result = ReadR1MongoQualificationResult(
        evidence_count=_EVIDENCE_COUNT,
        page_size=_PAGE_SIZE,
        first_page_canonical_gets=first_gets,
        first_page_index_queries=first_queries,
        latencies_ms=tuple(latencies),
        union_count=len(union),
        analyzer_fidelity_percent=100.0 if len(union) == _EVIDENCE_COUNT else 0.0,
        passed=passed,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "mongo-qualification.json").write_text(
        json.dumps(
            {
                **asdict(result),
                "latencies_ms": [asdict(item) for item in result.latencies_ms],
                "latency_p50_ms": statistics.median(
                    [item.latency_ms for item in result.latencies_ms],
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    inner.close()
    return result


if __name__ == "__main__":
    outcome = run_read_r1_mongo_qualification(
        artifact_dir=Path(".tmp/session/diag-functional-read-r1"),
    )
    print(json.dumps(asdict(outcome), indent=2, default=str))
    raise SystemExit(0 if outcome.passed else 1)
