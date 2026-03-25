# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import pytest


from pydantic import BaseModel

from intergrax.runtime.tools.sqlite_idempotency_store import (
    SQLiteIdempotencyStore,
)
from intergrax.contracts.idempotency_store import InvocationStatus
from intergrax.tools.execution_models import ToolExecutionResult
from testing_support.builder import prepare_sqlite_db

pytestmark = pytest.mark.integration


class DummyOutput(BaseModel):
    value: int


def test_sqlite_idempotency_restart_safe() -> None:
    tenant_id = "tenant_A"
    key = "key_123"

    db_path = prepare_sqlite_db("ledger.db")
    
    # --- First process ---
    store1 = SQLiteIdempotencyStore(str(db_path))

    result = ToolExecutionResult(
        success=True,
        output=DummyOutput(value=42),
        error=None,
    )

    store1.record_started(tenant_id, key)
    store1.record_completed(tenant_id, key, result)

    assert store1.get_status(tenant_id, key) == InvocationStatus.COMPLETED

    # --- Simulate restart ---
    store2 = SQLiteIdempotencyStore(str(db_path))

    status_after_restart = store2.get_status(tenant_id, key)
    assert status_after_restart == InvocationStatus.COMPLETED

    restored = store2.get_completed_result(tenant_id, key)

    assert restored is not None
    assert restored.success is True
    assert restored.output.value == 42
    assert restored.error is None
