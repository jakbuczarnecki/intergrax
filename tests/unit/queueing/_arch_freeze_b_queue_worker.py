# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
    DocumentStoreTaskQueue,
    TaskRowClaimContext,
)
from intergrax.queueing.providers.document_store.task_row_lease import read_claim_from_row
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="claim")
    args = parser.parse_args()
    db = os.environ.get("ARCH_FREEZE_B_DB")
    if not db:
        raise SystemExit("ARCH_FREEZE_B_DB is required")
    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    store = SqliteFileDocumentStore(Path(db), cursor_secret=b"arch-freeze-b-queue-worker-secret32")
    queue = DocumentStoreTaskQueue(store)
    if args.mode == "seed":
        queue.enqueue(
            TaskRequest(
                tenant_id="tenant-a",
                run_id="r1",
                task_name="demo.echo",
                payload=b"{}",
                idempotency_key="claim-once",
            )
        )
        print("seeded")
        return 0
    if args.mode == "complete-without-claim":
        task_id = os.environ.get("ARCH_FREEZE_B_TASK_ID", "").strip()
        tenant_id = os.environ.get("ARCH_FREEZE_B_TENANT", "").strip()
        if not task_id or not tenant_id:
            raise SystemExit("ARCH_FREEZE_B_TASK_ID and ARCH_FREEZE_B_TENANT are required")
        handle = TaskHandle(
            task_id=task_id,
            provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
            tenant_id=tenant_id,
        )
        try:
            queue.mark_succeeded(handle, output=b"{}")
        except StaleClaimError:
            print("STALE")
            return 0
        print("SETTLED")
        return 1
    claimed = queue.claim_pending(
        limit=1,
        claim_context=TaskRowClaimContext(owner_id=f"pid-{os.getpid()}", lease_seconds=30),
    )
    if not claimed:
        print("0")
        return 0
    handle, _request = claimed[0]
    record = store.get("intergrax.task_queue.v1:tenant-a", handle.task_id)
    if record is None:
        print("0")
        return 1
    parsed = read_claim_from_row(record.data)
    if parsed is None:
        print("0")
        return 1
    ownership, _ledger_key = parsed
    queue.mark_succeeded(handle, output=b"{}", presented_claim=ownership)
    print("1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
