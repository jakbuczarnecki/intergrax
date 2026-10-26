# © Artur Czarnecki. All rights reserved.

"""RabbitMQ M worker process for DIST-SCALE-X P2 L4."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from intergrax.integrations.providers.message_bus.rabbitmq.bundle import (
    create_rabbitmq_integration,
    create_rabbitmq_worker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from tests.qualification.dist_scale_x._p2_support import (
    P2_RABBIT_HOST,
    P2_RABBIT_PASS,
    P2_RABBIT_USER,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str):
        return self._data.get((tenant_id, key))

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds=None) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected,
        new_value: bytes,
        *,
        ttl_seconds=None,
    ) -> bool:
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class _Out(BaseModel):
    ok: bool = True


def _append_completion(metrics_dir: Path | None, run_id: str) -> None:
    if metrics_dir is None:
        return
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"completions_{os.getpid()}.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{run_id}\n")
        handle.flush()


def poll_loop(
    queue_name: str,
    stop_idle_s: float,
    *,
    metrics_dir: Path | None = None,
) -> int:
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ):
        _ = tenant_id, payload, idempotency_key, execution_identity
        _append_completion(metrics_dir, run_id)
        return ToolExecutionResult(success=True, output=_Out(), error=None)

    registry.register("dist_scale_x.p2.rabbit", handler)
    kv = InMemoryKVStore()
    bundle = create_rabbitmq_integration(
        kv_store=kv,
        host=P2_RABBIT_HOST,
        username=P2_RABBIT_USER,
        password=P2_RABBIT_PASS,
        queue=queue_name,
    )
    worker = create_rabbitmq_worker(
        kv_store=kv,
        execution_registry=registry,
        host=P2_RABBIT_HOST,
        username=P2_RABBIT_USER,
        password=P2_RABBIT_PASS,
        queue=queue_name,
        consumer=bundle.consumer,
        poll_timeout_seconds=0.3,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    processed = 0
    idle_start: float | None = None
    while True:
        raw = bundle.consumer.poll(timeout_seconds=0.3)
        if raw is not None:
            worker.process_message(raw_payload=raw)
            processed += 1
            idle_start = None
            continue
        if idle_start is None:
            idle_start = time.monotonic()
        elif time.monotonic() - idle_start >= stop_idle_s:
            break
    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default=f"dist-scale-x-p2-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--stop-idle-s", type=float, default=2.0)
    parser.add_argument("--metrics-dir", default="")
    args = parser.parse_args()
    metrics = Path(args.metrics_dir) if args.metrics_dir.strip() else None
    processed = 0
    try:
        processed = poll_loop(args.queue, args.stop_idle_s, metrics_dir=metrics)
    finally:
        if metrics is not None:
            metrics.mkdir(parents=True, exist_ok=True)
            (metrics / f"worker_{os.getpid()}.json").write_text(
                json.dumps({"processed": processed}),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
