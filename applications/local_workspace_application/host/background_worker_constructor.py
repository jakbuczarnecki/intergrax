# © Artur Czarnecki. All rights reserved.

"""Typed background worker construction seam for LKW product composition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_worker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence


@runtime_checkable
class BlockingBackgroundWorker(Protocol):
    def start(self) -> None: ...


class BackgroundWorkerConstructor(Protocol):
    def __call__(
        self,
        *,
        kv_store: DistributedKVStore,
        execution_registry: TaskExecutionRegistry,
        idempotency_store: IdempotencyStore | None,
        causal_evidence_persistence: CausalEvidencePersistence,
    ) -> BlockingBackgroundWorker: ...


def create_default_background_worker(
    *,
    kv_store: DistributedKVStore,
    execution_registry: TaskExecutionRegistry,
    idempotency_store: IdempotencyStore | None,
    causal_evidence_persistence: CausalEvidencePersistence,
) -> BlockingBackgroundWorker:
    worker = create_kafka_worker(
        kv_store=kv_store,
        execution_registry=execution_registry,
        idempotency_store=idempotency_store,
        causal_evidence_persistence=causal_evidence_persistence,
    )
    if not isinstance(worker, BlockingBackgroundWorker):
        raise TypeError("create_kafka_worker returned an invalid worker type")
    return worker
