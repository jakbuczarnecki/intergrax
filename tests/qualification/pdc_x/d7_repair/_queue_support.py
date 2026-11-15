# © Artur Czarnecki. All rights reserved.

"""Helpers for the DocumentStoreTaskQueue repair tests (F-D7-06, 07, 08)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryPageV1,
    DocumentRecord,
)
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.worker.retry_policy import RetryPolicy

TENANT = "t-d7q"
TASK = "pdc.d7.repair"


def request(*, key: str | None = None, tenant: str = TENANT) -> TaskRequest:
    return TaskRequest(tenant_id=tenant, run_id="r-d7q", task_name=TASK, payload=b"{}", idempotency_key=key)


def backoff_policy(*, max_retries: int = 5, seconds: float = 3600.0) -> RetryPolicy:
    return RetryPolicy(
        max_retries=max_retries,
        initial_backoff_seconds=seconds,
        backoff_multiplier=1.0,
        max_backoff_seconds=None,
        jitter=False,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=True,
    )


def claim_one(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> None:
    claimed = queue.claim_pending(tenant_id=handle.tenant_id, limit=1000)
    assert handle.task_id in {h.task_id for h, _ in claimed}


class SlowReadStore(InMemoryDocumentStore):
    """In-memory conditional store whose reads are slow, to widen read-modify-write races."""

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        found = super().get(partition_key, row_key)
        time.sleep(0.0005)
        return found


class PlainStore:
    """A DocumentStore without the conditional-write capability (plain contract only)."""

    def __init__(self) -> None:
        self._inner = InMemoryDocumentStore()

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._inner.get(partition_key, row_key)

    def put(self, document: DocumentRecord) -> None:
        self._inner.put(document)

    def delete(self, partition_key: str, row_key: str) -> None:
        self._inner.delete(partition_key, row_key)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        return self._inner.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
            row_key_upper_bound=row_key_upper_bound,
            data_equalities=data_equalities,
            sort=sort,
        )

    def close(self) -> None:
        self._inner.close()


def raw(record: DocumentRecord) -> dict[str, Any]:
    return dict(record.data)
