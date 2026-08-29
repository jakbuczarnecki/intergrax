# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType, parse_runtime_event_payload
from intergrax.runtime.events.spine_consolidation import (
    LEGACY_SPINE_TO_PLATFORM_KIND,
    PLATFORM_KIND_CATALOG,
    assert_publication_spine_budget,
    build_platform_signal_event,
    migrate_legacy_spine_payload,
    publication_spine_type_count,
    should_persist_platform_kind,
)

pytestmark = pytest.mark.gate


def test_publication_spine_budget_within_target() -> None:
    assert publication_spine_type_count() <= 56
    assert_publication_spine_budget()


def test_legacy_mapping_covers_platform_catalog() -> None:
    for entry in PLATFORM_KIND_CATALOG.values():
        assert LEGACY_SPINE_TO_PLATFORM_KIND[entry.legacy_spine_value] == entry.kind


def test_build_platform_signal_event_emits_domain_signal() -> None:
    run_id = mint_run_id()
    task_id = mint_task_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        event = build_platform_signal_event(
            kind="platform.hook.hook_timeout",
            task_id=task_id,
            run_id=run_id,
            phase=ExecutionPhase.INTAKE,
            payload={"hook_name": "demo"},
        )
    finally:
        reset_active_execution_identity(token)
    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == "platform.hook.hook_timeout"
    assert event.ops_hint == "ops:alert"
    assert event.payload["legacy_spine_type"] == "hook_timeout"
    assert event.payload["hook_name"] == "demo"


def test_migrate_legacy_spine_payload_on_read() -> None:
    raw = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "schema_version": "runtime_event.v2",
        "event_type": "scale_failed",
        "phase": ExecutionPhase.STEP_EXECUTION,
        "payload": {"reason": "denied"},
    }
    migrated = migrate_legacy_spine_payload(raw)
    assert migrated["event_type"] == RuntimeEventType.DOMAIN_SIGNAL
    assert migrated["event_kind"] == "platform.capacity.scale_failed"
    assert migrated["payload"]["legacy_spine_type"] == "scale_failed"
    event = parse_runtime_event_payload(migrated)
    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == "platform.capacity.scale_failed"


def test_platform_kind_sampling_deterministic() -> None:
    kind = "platform.capacity.capacity_signal_collected"
    first = should_persist_platform_kind(kind, "evt-a")
    second = should_persist_platform_kind(kind, "evt-a")
    assert first == second
