# © Artur Czarnecki. All rights reserved.

"""Qualification-owned contracts for DG-001B R5-R1 bootstrap failure injection."""

from __future__ import annotations

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
from local_workspace_application.host.background_worker_constructor import (
    BackgroundWorkerConstructor,
    BlockingBackgroundWorker,
)

_QUALIFICATION_SECRET_SENTINEL = "DG001B-R5-R1-SECRET-SENTINEL"
_TYPED_BOOTSTRAP_EXCEPTION_MESSAGE = (
    f"create_kafka_worker composition failure {_QUALIFICATION_SECRET_SENTINEL}"
)


def qualification_secret_sentinel() -> str:
    """Return the qualification-only sentinel used in thrown bootstrap exceptions."""
    return _QUALIFICATION_SECRET_SENTINEL


class ControlledFailingBackgroundWorkerConstructor:
    """Inject a deterministic B6 composition failure through the generic worker seam."""

    def __call__(
        self,
        *,
        kv_store: DistributedKVStore,
        execution_registry: TaskExecutionRegistry,
        idempotency_store: IdempotencyStore | None,
        causal_evidence_persistence: CausalEvidencePersistence,
    ) -> BlockingBackgroundWorker:
        del kv_store, execution_registry, idempotency_store, causal_evidence_persistence
        raise TypeError(_TYPED_BOOTSTRAP_EXCEPTION_MESSAGE)


def controlled_failing_background_worker_constructor() -> BackgroundWorkerConstructor:
    return ControlledFailingBackgroundWorkerConstructor()
