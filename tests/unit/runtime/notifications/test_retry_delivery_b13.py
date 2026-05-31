# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.deliveries.delivery_ledger import InMemoryDeliveryLedger
from intergrax.runtime.notifications.deliveries.retry_delivery import RetryingNotificationDelivery

pytestmark = pytest.mark.gate


class _FlakyDelivery(NotificationDelivery):
    def __init__(self, *, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self.calls = 0

    async def deliver(self, *, destination: str, payload, headers=None) -> None:
        _ = destination, headers
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("transport down")


@pytest.mark.asyncio
async def test_retry_delivery_succeeds_after_transient_failure():
    inner = _FlakyDelivery(fail_times=1)
    ledger = InMemoryDeliveryLedger()
    delivery = RetryingNotificationDelivery(inner, max_attempts=3, ledger=ledger)
    receipt = await delivery.deliver(
        destination="https://example.test/hook",
        payload={"task_id": "task_1", "title": "hello"},
    )
    assert inner.calls == 2
    assert receipt is not None
    assert receipt.status == "delivered"
    assert len(ledger.list_receipts()) == 1


@pytest.mark.asyncio
async def test_retry_delivery_records_dead_letter_after_exhaustion():
    inner = _FlakyDelivery(fail_times=5)
    ledger = InMemoryDeliveryLedger()
    delivery = RetryingNotificationDelivery(inner, max_attempts=2, ledger=ledger)
    receipt = await delivery.deliver(
        destination="https://example.test/hook",
        payload={"task_id": "task_2"},
    )
    assert inner.calls == 2
    assert receipt is not None
    assert receipt.status == "dead_letter"
    assert len(ledger.list_dead_letters()) == 1
