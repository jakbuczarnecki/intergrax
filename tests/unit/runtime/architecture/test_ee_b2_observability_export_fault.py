# © Artur Czarnecki. All rights reserved.

"""EE-B2 — OTLP/export fault isolation from authoritative execution evidence."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.export_wiring import (
    make_observability_export_runtime_plugin,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FailingExporter:
    def __init__(self) -> None:
        self.calls = 0

    async def export(self, envelope) -> None:
        self.calls += 1
        raise RuntimeError("otlp_sink_down")


@pytest.mark.asyncio
async def test_ee_b2_otlp_failure_does_not_remove_canonical_event() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=False)
    identity = runtime_event_test_identity()
    exporter = _FailingExporter()
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())
    event = sample_runtime_event(tenant_id="tenant-a", **identity)  # type: ignore[arg-type]
    await bus.publish(event)
    persisted = store.list_for_run(identity["run_id"], tenant_id="tenant-a")
    assert len(persisted) == 1
    assert exporter.calls >= 1


@pytest.mark.asyncio
async def test_ee_b2_observability_export_failure_not_mandatory_evidence_failure() -> (
    None
):
    exporter = _FailingExporter()
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-ee-b2",
    )
    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    assert result.exported is False
    assert result.reason == "exporter_failed"
