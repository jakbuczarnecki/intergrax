# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from intergrax.queueing.contracts.task_queue import TaskRequest
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
    TaskRowClaimContext,
)
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
    claimed = queue.claim_pending(
        limit=1,
        claim_context=TaskRowClaimContext(owner_id=f"pid-{os.getpid()}", lease_seconds=30),
    )
    if not claimed:
        print("0")
        return 0
    handle, _request = claimed[0]
    queue.mark_succeeded(handle, output=b"{}")
    print("1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
