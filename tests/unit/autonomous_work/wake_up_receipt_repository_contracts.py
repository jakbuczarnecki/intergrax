# © Artur Czarnecki. All rights reserved.

"""Shared wake-up receipt repository contract suite."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable

from intergrax.autonomous_work.repository import (
    WorkerWakeUpReceiptClaimStatus,
    WorkerWakeUpReceiptRepository,
)
from intergrax.contracts.autonomous_work import (
    WakeUpId,
    WorkerInstanceId,
    WorkerWakeUpReceipt,
    WorkerWakeUpSourceKind,
    mint_wake_up_id,
    mint_worker_instance_id,
    wake_up_signals_logically_equivalent,
)
from intergrax.contracts.autonomous_work.references import (
    WakeUpCorrelationRef,
    WakeUpSourceRef,
)

_UTC = timezone.utc


def wake_up_receipt(
    *,
    worker_instance_id: WorkerInstanceId | None = None,
    wake_up_id: WakeUpId | None = None,
    accepted_at: datetime | None = None,
    source_kind: WorkerWakeUpSourceKind = WorkerWakeUpSourceKind.EXTERNAL_EVENT,
    source_ref: WakeUpSourceRef | None = None,
    occurred_at: datetime | None = None,
    delivery_identity: WakeUpId | None = None,
    correlation_ref: WakeUpCorrelationRef | None = None,
) -> WorkerWakeUpReceipt:
    now = accepted_at or datetime(2026, 9, 3, 12, 0, tzinfo=_UTC)
    wake_id = wake_up_id or mint_wake_up_id()
    return WorkerWakeUpReceipt(
        worker_instance_id=worker_instance_id or mint_worker_instance_id(),
        wake_up_id=wake_id,
        source_kind=source_kind,
        source_ref=source_ref or WakeUpSourceRef("source/external/event-1"),
        occurred_at=occurred_at or datetime(2026, 9, 3, 11, 59, tzinfo=_UTC),
        accepted_at=now,
        delivery_identity=delivery_identity or wake_id,
        correlation_ref=correlation_ref,
    )


def run_wake_up_receipt_contract_suite(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    receipt = wake_up_receipt()
    claim = repo.claim(receipt)
    assert claim.status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    assert claim.receipt == receipt

    duplicate = repo.claim(receipt)
    assert duplicate.status is WorkerWakeUpReceiptClaimStatus.DUPLICATE
    assert duplicate.receipt == receipt

    loaded = repo.get(
        worker_instance_id=receipt.worker_instance_id,
        wake_up_id=receipt.wake_up_id,
    )
    assert loaded == receipt


def test_wake_up_receipt_claim_is_idempotent(factory: Callable[[], WorkerWakeUpReceiptRepository]) -> None:
    run_wake_up_receipt_contract_suite(factory)


def test_wake_up_receipt_distinct_events_both_accepted(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    worker_id = mint_worker_instance_id()
    first = wake_up_receipt(worker_instance_id=worker_id)
    second = wake_up_receipt(worker_instance_id=worker_id)
    assert repo.claim(first).status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    assert repo.claim(second).status is WorkerWakeUpReceiptClaimStatus.CLAIMED


def test_wake_up_receipt_cross_worker_isolation(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    wake_id = mint_wake_up_id()
    worker_a = mint_worker_instance_id()
    worker_b = mint_worker_instance_id()
    receipt_a = wake_up_receipt(worker_instance_id=worker_a, wake_up_id=wake_id)
    receipt_b = wake_up_receipt(worker_instance_id=worker_b, wake_up_id=wake_id)
    assert repo.claim(receipt_a).status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    assert repo.claim(receipt_b).status is WorkerWakeUpReceiptClaimStatus.CLAIMED


def test_wake_up_receipt_conflicting_source_kind(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    worker_id = mint_worker_instance_id()
    wake_id = mint_wake_up_id()
    first = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        source_kind=WorkerWakeUpSourceKind.QUEUE_DELIVERY,
        source_ref=WakeUpSourceRef("queue/order-123"),
    )
    second = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        source_kind=WorkerWakeUpSourceKind.OPERATOR,
        source_ref=WakeUpSourceRef("operator/manual"),
    )
    assert repo.claim(first).status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    conflict = repo.claim(second)
    assert conflict.status is WorkerWakeUpReceiptClaimStatus.CONFLICT
    assert conflict.receipt == first


def test_wake_up_receipt_conflicting_source_ref(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    worker_id = mint_worker_instance_id()
    wake_id = mint_wake_up_id()
    first = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        source_ref=WakeUpSourceRef("queue/order-123"),
    )
    second = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        source_ref=WakeUpSourceRef("queue/order-456"),
    )
    assert repo.claim(first).status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    assert repo.claim(second).status is WorkerWakeUpReceiptClaimStatus.CONFLICT


def test_wake_up_receipt_conflicting_correlation(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    worker_id = mint_worker_instance_id()
    wake_id = mint_wake_up_id()
    first = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        correlation_ref=WakeUpCorrelationRef("corr/a"),
    )
    second = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        correlation_ref=WakeUpCorrelationRef("corr/b"),
    )
    assert repo.claim(first).status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    assert repo.claim(second).status is WorkerWakeUpReceiptClaimStatus.CONFLICT


def test_wake_up_receipt_different_delivery_identity_is_duplicate(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    worker_id = mint_worker_instance_id()
    wake_id = mint_wake_up_id()
    first = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        delivery_identity=mint_wake_up_id(),
    )
    second = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        delivery_identity=mint_wake_up_id(),
    )
    assert wake_up_signals_logically_equivalent(first, second)
    assert repo.claim(first).status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    duplicate = repo.claim(second)
    assert duplicate.status is WorkerWakeUpReceiptClaimStatus.DUPLICATE
    assert duplicate.receipt == first


def test_wake_up_receipt_concurrent_duplicate_one_wins(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    receipt = wake_up_receipt()
    outcomes: list[WorkerWakeUpReceiptClaimStatus] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        repo = factory()
        barrier.wait(timeout=5)
        outcomes.append(repo.claim(receipt).status)

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(WorkerWakeUpReceiptClaimStatus.CLAIMED) == 1
    assert outcomes.count(WorkerWakeUpReceiptClaimStatus.DUPLICATE) == 1


def test_wake_up_receipt_concurrent_conflicting_one_claimed_one_conflict(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    worker_id = mint_worker_instance_id()
    wake_id = mint_wake_up_id()
    first = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        source_kind=WorkerWakeUpSourceKind.QUEUE_DELIVERY,
        source_ref=WakeUpSourceRef("queue/order-123"),
    )
    second = wake_up_receipt(
        worker_instance_id=worker_id,
        wake_up_id=wake_id,
        source_kind=WorkerWakeUpSourceKind.OPERATOR,
        source_ref=WakeUpSourceRef("operator/manual"),
    )
    outcomes: list[WorkerWakeUpReceiptClaimStatus] = []
    barrier = threading.Barrier(2)

    def attempt(receipt: WorkerWakeUpReceipt) -> None:
        repo = factory()
        barrier.wait(timeout=5)
        outcomes.append(repo.claim(receipt).status)

    threads = [
        threading.Thread(target=attempt, args=(first,)),
        threading.Thread(target=attempt, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(WorkerWakeUpReceiptClaimStatus.CLAIMED) == 1
    assert outcomes.count(WorkerWakeUpReceiptClaimStatus.CONFLICT) == 1
