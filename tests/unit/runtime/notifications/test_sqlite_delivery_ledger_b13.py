# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.notifications.deliveries.sqlite_delivery_ledger import SQLiteDeliveryLedger

pytestmark = pytest.mark.gate


def test_sqlite_delivery_ledger_persists_dead_letters(tmp_path):
    ledger = SQLiteDeliveryLedger(db_path=tmp_path / "delivery.db")
    receipt = ledger.record_dead_letter(
        destination="https://example.test/hook",
        task_id="task_1",
        channel="webhook",
        attempts=3,
        last_error="timeout",
        payload_summary={"task_id": "task_1"},
    )
    assert receipt.status == "dead_letter"

    reloaded = SQLiteDeliveryLedger(db_path=tmp_path / "delivery.db")
    dead_letters = reloaded.list_dead_letters(limit=10)
    assert len(dead_letters) == 1
    assert dead_letters[0].task_id == "task_1"
    assert dead_letters[0].last_error == "timeout"

    receipts = reloaded.list_receipts(limit=10)
    assert receipts == []

    ledger.record_success(
        destination="https://example.test/hook",
        task_id="task_2",
        channel="webhook",
        attempts=1,
    )
    assert len(reloaded.list_receipts(limit=10)) == 1
