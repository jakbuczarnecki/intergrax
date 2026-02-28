# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid

import pytest
from redis import Redis
from pydantic import BaseModel

from intergrax.contracts.idempotency_store import InvocationStatus
from intergrax.distributed.providers.redis_idempotency_store import (
    RedisIdempotencyStore,
)
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = pytest.mark.integration


class DummyOutput(BaseModel):
    value: int


def _create_store() -> RedisIdempotencyStore:
    redis = Redis(host="localhost", port=6379, decode_responses=False)
    return RedisIdempotencyStore(redis_client=redis)


def test_full_ledger_flow() -> None:
    store = _create_store()

    tenant = "tenant_A"
    key = f"test:{uuid.uuid4()}"

    # STARTED
    store.record_started(tenant, key)

    status = store.get_status(tenant, key)
    assert status == InvocationStatus.STARTED

    # COMPLETED
    result = ToolExecutionResult(
        success=True,
        output=DummyOutput(value=42),
        error=None,
    )

    store.record_completed(tenant, key, result)

    status_after = store.get_status(tenant, key)
    assert status_after == InvocationStatus.COMPLETED

    replay = store.get_completed_result(tenant, key)
    assert replay is not None
    assert replay.success is True
    assert replay.output.value == 42


def test_duplicate_started_fails() -> None:
    store = _create_store()

    tenant = "tenant_B"
    key = f"test:{uuid.uuid4()}"

    store.record_started(tenant, key)

    with pytest.raises(RuntimeError, match="Invocation already exists"):
        store.record_started(tenant, key)



def test_invalid_transition_fails() -> None:
    store = _create_store()

    tenant = "tenant_C"
    key = f"test:{uuid.uuid4()}"

    store.record_started(tenant, key)

    result = ToolExecutionResult(
        success=True,
        output=DummyOutput(value=1),
        error=None,
    )

    store.record_completed(tenant, key, result)

    with pytest.raises(RuntimeError, match="Invalid state transition"):
        store.record_completed(tenant, key, result)
