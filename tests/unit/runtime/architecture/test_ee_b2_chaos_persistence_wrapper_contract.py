# © Artur Czarnecki. All rights reserved.

"""EE-B2 — FailOnAppendPersistence contract fidelity (return forwarding + fault order)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.chaos.failing_persistence import FailOnAppendPersistence
from testing_support.chaos.fault_plan import FailOnCall

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b2_chaos_persistence_append_forwards_inner_result() -> None:
    inner = MagicMock(spec=RuntimeEventPersistence)
    expected = MagicMock(name="positioned_runtime_event")
    inner.append.return_value = expected
    wrapper = FailOnAppendPersistence(
        inner,
        fail_on=FailOnCall(call_number=99, message="not_this_call"),
    )
    event = sample_runtime_event(tenant_id="tenant-wrap")
    result = wrapper.append(event, tenant_id="tenant-wrap")
    assert result is expected
    inner.append.assert_called_once_with(event, tenant_id="tenant-wrap")


def test_ee_b2_chaos_persistence_fail_before_inner_append() -> None:
    inner = MagicMock(spec=RuntimeEventPersistence)
    wrapper = FailOnAppendPersistence(
        inner,
        fail_on=FailOnCall(call_number=1, message="append_fault"),
    )
    event = sample_runtime_event(tenant_id="tenant-fail")
    with pytest.raises(RuntimeError, match="append_fault"):
        wrapper.append(event, tenant_id="tenant-fail")
    inner.append.assert_not_called()


def test_ee_b2_chaos_persistence_fail_on_nth_call_order() -> None:
    inner = MagicMock(spec=RuntimeEventPersistence)
    inner.append.return_value = MagicMock(name="positioned")
    wrapper = FailOnAppendPersistence(
        inner,
        fail_on=FailOnCall(call_number=2, message="second_call_fault"),
    )
    event = sample_runtime_event(tenant_id="tenant-order")
    wrapper.append(event, tenant_id="tenant-order")
    with pytest.raises(RuntimeError, match="second_call_fault"):
        wrapper.append(event, tenant_id="tenant-order")
    assert inner.append.call_count == 1
