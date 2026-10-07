# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R1 tenant lifecycle and recorder binding gates."""

from __future__ import annotations

import pytest

from intergrax.runtime.events.active_runtime_event_recorder import (
    ActiveRuntimeEventRecorderBinding,
    bind_active_runtime_event_recorder,
    peek_active_runtime_event_recorder,
    peek_active_runtime_event_tenant_id,
    reset_active_runtime_event_recorder,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp4r1_q04_recorder_binding_resets_tenant() -> None:
    bus = RuntimeEventBus(record_history=True)
    binding = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
    assert isinstance(binding, ActiveRuntimeEventRecorderBinding)
    assert peek_active_runtime_event_tenant_id() == "tenant-a"
    assert peek_active_runtime_event_recorder() is bus
    reset_active_runtime_event_recorder(binding)
    assert peek_active_runtime_event_tenant_id() == ""
    assert peek_active_runtime_event_recorder() is None


def test_txp4r1_q05_nested_tenant_binding() -> None:
    outer = bind_active_runtime_event_recorder(None, tenant_id="tenant-a")
    inner = bind_active_runtime_event_recorder(None, tenant_id="tenant-b")
    assert peek_active_runtime_event_tenant_id() == "tenant-b"
    reset_active_runtime_event_recorder(inner)
    assert peek_active_runtime_event_tenant_id() == "tenant-a"
    reset_active_runtime_event_recorder(outer)
    assert peek_active_runtime_event_tenant_id() == ""


def test_txp4r1_q06_sequential_tenant_isolation() -> None:
    first = bind_active_runtime_event_recorder(None, tenant_id="tenant-a")
    reset_active_runtime_event_recorder(first)
    second = bind_active_runtime_event_recorder(None, tenant_id="tenant-b")
    assert peek_active_runtime_event_tenant_id() == "tenant-b"
    reset_active_runtime_event_recorder(second)
    assert peek_active_runtime_event_tenant_id() == ""


def test_txp4r1_ten_l4_stale_tenant_cleared_without_recorder() -> None:
    stale = bind_active_runtime_event_recorder(None, tenant_id="tenant-stale")
    reset_active_runtime_event_recorder(stale)
    without_recorder = bind_active_runtime_event_recorder(None, tenant_id="tenant-fresh")
    assert peek_active_runtime_event_tenant_id() == "tenant-fresh"
    reset_active_runtime_event_recorder(without_recorder)
    assert peek_active_runtime_event_tenant_id() == ""
