# © Artur Czarnecki. All rights reserved.

"""AW-4A — in-memory wake-up receipt repository contract tests."""

from __future__ import annotations

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerWakeUpReceiptRepository
from tests.unit.autonomous_work import wake_up_receipt_repository_contracts as contract_suite

pytestmark = pytest.mark.unit


def test_in_memory_wake_up_receipt_contract_suite() -> None:
    contract_suite.run_wake_up_receipt_contract_suite(InMemoryWorkerWakeUpReceiptRepository)


def test_in_memory_wake_up_receipt_distinct_events() -> None:
    contract_suite.test_wake_up_receipt_distinct_events_both_accepted(
        InMemoryWorkerWakeUpReceiptRepository
    )


def test_in_memory_wake_up_receipt_cross_worker_isolation() -> None:
    contract_suite.test_wake_up_receipt_cross_worker_isolation(
        InMemoryWorkerWakeUpReceiptRepository
    )


def test_in_memory_wake_up_receipt_concurrent_duplicate() -> None:
    shared_repo = InMemoryWorkerWakeUpReceiptRepository()

    def factory() -> InMemoryWorkerWakeUpReceiptRepository:
        return shared_repo

    contract_suite.test_wake_up_receipt_concurrent_duplicate_one_wins(factory)
