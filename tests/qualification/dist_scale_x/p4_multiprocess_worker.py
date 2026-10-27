# © Artur Czarnecki. All rights reserved.

"""OS-process WL-5 writer (DIST-SCALE-X P4-A) — gate-driven aggregate rate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tests.qualification.dist_scale_x.p4_persistence_op import (
    execute_p4_persistence_op,
    mint_qual_task_id,
)
from tests.qualification.dist_scale_x.p4_wl5_gate import Wl5RateGate


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
    worker_id: str,
    worker_index: int,
    gate_db: Path,
    writers: int,
) -> None:
    gate = Wl5RateGate(gate_db)
    last_write_task_id = ""
    writes = 0
    reads = 0
    errors = 0
    pad = _pad(payload_bytes)
    try:
        while not stop_file.exists():
            seq = gate.try_claim(worker_id)
            if seq is None:
                time.sleep(0.005)
                continue
            op_index = seq * writers + worker_index
            is_read = op_index % 10 == 0
            write_task_id = None if is_read else mint_qual_task_id()
            read_task_id = last_write_task_id if is_read else None
            result = execute_p4_persistence_op(
                dsn=dsn,
                schema=schema,
                tenant_id=tenant_id,
                op_index=op_index,
                payload_pad=pad,
                task_id=read_task_id if is_read else write_task_id,
            )
            if result.error:
                errors += 1
            elif result.kind == "read":
                reads += 1
            elif result.kind == "write":
                writes += 1
                last_write_task_id = result.task_id
    except Exception as exc:  # noqa: BLE001
        metrics_file.write_text(
            json.dumps({"worker_id": worker_id, "fatal": f"{type(exc).__name__}:{exc}"}),
            encoding="utf-8",
        )
        raise
    metrics_file.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
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
    parser.add_argument("--gate-db", required=True)
    parser.add_argument("--writers", type=int, default=2)
    args = parser.parse_args()
    worker_loop(
        dsn=args.dsn,
        schema=args.schema,
        tenant_id=args.tenant_id,
        payload_bytes=args.payload_bytes,
        stop_file=Path(args.stop_file),
        metrics_file=Path(args.metrics_file),
        worker_id=f"w{args.worker_index}",
        worker_index=args.worker_index,
        gate_db=Path(args.gate_db),
        writers=args.writers,
    )


if __name__ == "__main__":
    main()
