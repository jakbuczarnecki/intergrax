# © Artur Czarnecki. All rights reserved.

"""Shared wake-up receipt repository contract suite."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable

from intergrax.autonomous_work.repository import WorkerWakeUpReceiptRepository
from intergrax.contracts.autonomous_work import (
    WakeUpId,
    WorkerInstanceId,
    WorkerWakeUpReceipt,
    WorkerWakeUpSourceKind,
    mint_wake_up_id,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import WakeUpSourceRef

_UTC = timezone.utc


def wake_up_receipt(
    *,
    worker_instance_id: WorkerInstanceId | None = None,
    wake_up_id: WakeUpId | None = None,
    accepted_at: datetime | None = None,
) -> WorkerWakeUpReceipt:
    now = accepted_at or datetime(2026, 9, 3, 12, 0, tzinfo=_UTC)
    wake_id = wake_up_id or mint_wake_up_id()
    return WorkerWakeUpReceipt(
        worker_instance_id=worker_instance_id or mint_worker_instance_id(),
        wake_up_id=wake_id,
        source_kind=WorkerWakeUpSourceKind.EXTERNAL_EVENT,
        source_ref=WakeUpSourceRef("source/external/event-1"),
        occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=_UTC),
        accepted_at=now,
        delivery_identity=wake_id,
        correlation_ref=None,
    )


def run_wake_up_receipt_contract_suite(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    receipt = wake_up_receipt()
    claim = repo.claim(receipt)
    assert claim.duplicate is False
    assert claim.receipt == receipt

    duplicate = repo.claim(receipt)
    assert duplicate.duplicate is True
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
    assert repo.claim(first).duplicate is False
    assert repo.claim(second).duplicate is False


def test_wake_up_receipt_cross_worker_isolation(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    repo = factory()
    wake_id = mint_wake_up_id()
    worker_a = mint_worker_instance_id()
    worker_b = mint_worker_instance_id()
    receipt_a = wake_up_receipt(worker_instance_id=worker_a, wake_up_id=wake_id)
    receipt_b = wake_up_receipt(worker_instance_id=worker_b, wake_up_id=wake_id)
    assert repo.claim(receipt_a).duplicate is False
    assert repo.claim(receipt_b).duplicate is False


def test_wake_up_receipt_concurrent_duplicate_one_wins(
    factory: Callable[[], WorkerWakeUpReceiptRepository],
) -> None:
    receipt = wake_up_receipt()
    outcomes: list[bool] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        repo = factory()
        barrier.wait(timeout=5)
        outcomes.append(repo.claim(receipt).duplicate)

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(False) == 1
    assert outcomes.count(True) == 1
