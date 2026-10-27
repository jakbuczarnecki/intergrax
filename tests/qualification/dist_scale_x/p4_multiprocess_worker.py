# © Artur Czarnecki. All rights reserved.

"""OS-process WL-5 writer (DIST-SCALE-X P4-A)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tests.qualification.dist_scale_x.p4_persistence_op import (
    execute_p4_persistence_op,
    mint_qual_task_id,
)


def _pad(payload_bytes: int) -> str:
    return "x" * max(1, payload_bytes)


def worker_loop(
    *,
    dsn: str,
    schema: str,
    tenant_id: str,
    payload_bytes: int,
    stop_file: Path,
    metrics_file: Path,
    worker_index: int,
) -> None:
    op_index = worker_index
    task_ids: list[str] = []
    writes = 0
    reads = 0
    errors = 0
    while not stop_file.exists():
        tid = task_ids[-1] if task_ids and op_index % 10 == 0 else None
        if op_index % 10 != 0 or not task_ids:
            tid = mint_qual_task_id()
        result = execute_p4_persistence_op(
            dsn=dsn,
            schema=schema,
            tenant_id=tenant_id,
            op_index=op_index,
            payload_pad=_pad(payload_bytes),
            task_id=tid,
        )
        op_index += 2
        if result.error:
            errors += 1
        elif result.kind == "read":
            reads += 1
        elif result.kind == "write":
            writes += 1
            task_ids.append(result.task_id)
        time.sleep(0.001)
    metrics_file.write_text(
        json.dumps(
            {
                "worker_index": worker_index,
                "writes": writes,
                "reads": reads,
                "errors": errors,
            },
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    worker_loop(
        dsn=args.dsn,
        schema=args.schema,
        tenant_id=args.tenant_id,
        payload_bytes=args.payload_bytes,
        stop_file=Path(args.stop_file),
        metrics_file=Path(args.metrics_file),
        worker_index=args.worker_index,
    )


if __name__ == "__main__":
    main()
